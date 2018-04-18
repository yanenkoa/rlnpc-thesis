from collections import namedtuple
from typing import Tuple, NamedTuple, Callable, Optional, List, Deque

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Concatenate
from tensorflow import losses, Tensor
from tensorflow.python.training.saver import Saver

from trainer.GameObjects import World, PlayerMovementDirection, Player, SensedObject


Experiences = namedtuple("Experiences", [
    "states",
    "next_states",
    "action_indices",
    "rewards",
    "terminates"
])
# class Experiences(NamedTuple):
#     states: List[np.ndarray]
#     next_states: List[np.ndarray]
#     action_indices: np.ndarray
#     rewards: np.ndarray
#     terminates: np.ndarray


class EpisodeBuffer:
    """
    :type _buffer_size: int

    :type _array_size: int
    :type _current_ptr: int
    :type _indices_for_sampling: np.ndarray

    :type _state_buffers: List[np.ndarray]
    :type _next_states_buffer: List[np.ndarray]
    :type _action_index_buffer: np.ndarray
    :type _reward_buffer: np.ndarray
    :type _terminate_buffer: np.ndarray
    """

    def __init__(self, state_shapes: List[Tuple], buffer_size: int = 50000):
        self._buffer_size = buffer_size

        self._array_size = self._buffer_size * 2
        self._current_ptr = 0
        self._indices_for_sampling = np.arange(self._buffer_size)

        self._state_buffers = []
        self._next_states_buffer = []
        for shape in state_shapes:
            self._state_buffers.append(
                np.empty(
                    shape=(self._array_size, *shape),
                    dtype=np.float32
                )
            )
            self._next_states_buffer.append(
                np.empty(
                    shape=(self._array_size, *shape),
                    dtype=np.float32
                )
            )
        self._action_index_buffer = np.empty(
            shape=(self._array_size,),
            dtype=np.int32
        )
        self._reward_buffer = np.empty(
            shape=(self._array_size,),
            dtype=np.int32,
        )
        self._terminate_buffer = np.empty(
            shape=(self._array_size,),
            dtype=np.bool
        )

    def _reset_buffer(self, buffer: np.ndarray) -> None:
        buffer[0:self._buffer_size] = buffer[self._current_ptr - self._buffer_size: self._current_ptr]

    def _fill_buffer(self, buffer: np.ndarray, new_data: np.ndarray, n_experiences: int) -> None:
        buffer[self._current_ptr: self._current_ptr + n_experiences] = new_data

    def add_experiences(self, experiences: Experiences) -> None:
        states, next_states, action_indices, rewards, terminates = experiences
        n_experiences = rewards.size
        assert (
            np.all([state.shape[0] == n_experiences for state in states]) and
            np.all([next_state.shape[0] == n_experiences for next_state in next_states]) and
            action_indices.shape[0] == n_experiences and
            rewards.shape[0] == n_experiences and
            terminates.shape[0] == n_experiences
        )

        if self._current_ptr - 1 + n_experiences >= self._array_size:
            for state_buffer, next_state_buffer in zip(self._state_buffers, self._next_states_buffer):
                self._reset_buffer(state_buffer)
                self._reset_buffer(next_state_buffer)

            self._reset_buffer(self._action_index_buffer)
            self._reset_buffer(self._reward_buffer)
            self._reset_buffer(self._terminate_buffer)
            self._current_ptr = self._buffer_size

        for i in range(len(states)):
            self._fill_buffer(self._state_buffers[i], states[i], n_experiences)
            self._fill_buffer(self._next_states_buffer[i], next_states[i], n_experiences)

        self._fill_buffer(self._action_index_buffer, action_indices, n_experiences)
        self._fill_buffer(self._reward_buffer, action_indices, n_experiences)
        self._fill_buffer(self._terminate_buffer, action_indices, n_experiences)

        self._current_ptr += n_experiences

    def sample(self, sample_size: int) -> Experiences:
        assert self._current_ptr >= sample_size
        if self._current_ptr < self._buffer_size:
            indices = np.random.choice(
                self._current_ptr,
                size=sample_size,
                replace=False
            )
        else:
            indices = np.random.choice(
                self._current_ptr - self._buffer_size + self._indices_for_sampling,
                size=sample_size,
                replace=False
            )
        return Experiences(
            [buf[indices] for buf in self._state_buffers],
            [buf[indices] for buf in self._next_states_buffer],
            self._action_index_buffer[indices],
            self._reward_buffer[indices],
            self._terminate_buffer[indices]
        )


# LearningProcessConfig = namedtuple("LearningProcessConfig", [
#     "replay_size",
#     "update_frequency",
#     "reward_discount_coef",
#     "start_random_action_prob",
#     "end_random_action_prob",
#     "annealing_steps",
#     "n_training_episodes",
#     "pre_train_steps",
#     "max_ep_length",
#     "buffer_size",
# ])


LearningProcessConfig = namedtuple("LearningProcessConfig", [
    "replay_size",
    "update_frequency",
    "reward_discount_coef",
    "start_random_action_prob",
    "end_random_action_prob",
    "annealing_steps",
    "n_training_episodes",
    "pre_train_steps",
    "max_ep_length",
    "buffer_size",
])
# class LearningProcessConfig(NamedTuple):
#     replay_size = ...  # type: int
#     update_frequency = ...  # type: int
#     reward_discount_coef = ...  # type: float
#     start_random_action_prob = ...  # type: float
#     end_random_action_prob = ...  # type: float
#     annealing_steps = ...  # type: int
#     n_training_episodes = ...  # type: int
#     pre_train_steps = ...  # type: int
#     max_ep_length = ...  # type: int
#     buffer_size = ...  # type: int


class DeepQLearnerWithExperienceReplay:
    _game_world = ...  # type: World
    _output_angles = ...  # type: np.ndarray
    _session = ...  # type: tf.Session
    _time_between_actions_s = ...  # type: float
    _n_steps_back = ...  # type: int
    _process_config = ...  # type: LearningProcessConfig

    _player = ...  # type: Player
    _n_sensor_types = ...  # type: int
    _n_output_angles = ...  # type: int
    _n_actions = ...  # type: int
    _n_sensors = ...  # type: int
    _n_sensor_inputs = ...  # type: int
    _max_sensor_distance = ...  # type: float
    _sensor_state_shape = ...  # type: Tuple
    _heat_state_shape = ...  # type: Tuple
    _state_shapes = ...  # type: List[Tuple]

    _sensor_input_tensor = ...  # type: tf.Tensor
    _heat_input_tensor = ...  # type: tf.Tensor
    _position_input_tensor = ...  # type: tf.Tensor
    _action_index_tensor = ...  # type: tf.Tensor
    _chosen_actions_tensor = ...  # type: tf.Tensor
    _rewards_tensor = ...  # type: tf.Tensor
    _terminates_tensor = ...  # type: tf.Tensor
    _actions_qualities_tensor = ...  # type: tf.Tensor
    _replay_next_states_qualities_tensor = ...  # type: tf.Tensor
    _update_model = ...  # type: tf.Operation
    _saver = ...  # type: Saver

    def __init__(self,
                 world: World,
                 output_angles: np.ndarray,
                 session: tf.Session,
                 time_between_actions_s: float,
                 n_steps_back: int,
                 process_config: LearningProcessConfig):
        self._game_world = world
        self._output_angles = output_angles
        self._session = session
        self._time_between_actions_s = time_between_actions_s
        self._n_steps_back = n_steps_back
        self._process_config = process_config

        self._player = self._game_world.player
        # noinspection PyTypeChecker
        self._n_sensor_types = len(SensedObject) + 1
        self._n_output_angles = self._output_angles.size
        self._n_actions = self._n_output_angles + 1
        self._n_sensors = self._game_world.proximity_sensors_np.n_sensors
        self._n_sensor_inputs = self._n_sensors * self._n_sensor_types * self._n_steps_back
        self._max_sensor_distance = self._game_world.proximity_sensors_np.max_distance

        self._sensor_state_shape = (self._n_sensor_inputs,)
        self._heat_state_shape = (self._n_steps_back,)
        # self._angle_state_shape = (self._n_steps_back,)
        # self._position_state_shape = (2,)
        self._state_shapes = [
            self._sensor_state_shape,
            self._heat_state_shape,
            # self._angle_state_shape,
            # self._position_state_shape,
        ]

    def get_heat_input(self) -> np.ndarray:
        return np.array([self._player.heat])

    def _get_position_input(self) -> np.ndarray:
        return np.array([self._player.position.x, self._player.position.y])

    def _get_angle_movement(self, action_index: int) -> Tuple[float, PlayerMovementDirection]:
        if action_index == self._n_output_angles:
            return self._player.angle, PlayerMovementDirection.NONE
        else:
            return self._player.angle + self._output_angles[action_index], PlayerMovementDirection.FORWARD

    def _apply_action(self, action_index: int) -> None:
        angle, movement = self._get_angle_movement(action_index)
        self._game_world.update_player_angle(angle)
        self._game_world.move_player(self._time_between_actions_s, movement)
        self._game_world.update_state(self._time_between_actions_s)

    def get_sensor_input(self) -> np.ndarray:
        distances = self._game_world.proximity_sensors_np.distances
        object_types = self._game_world.proximity_sensors_np.object_types
        prox_sens_data = np.zeros(shape=(self._n_sensors, self._n_sensor_types), dtype=np.float32)
        prox_sens_data[:, 0] = distances / self._max_sensor_distance
        prox_sens_data[np.arange(0, self._n_sensors), object_types] = 1
        return prox_sens_data.flatten()

    def initialize(self) -> None:
        keras.backend.set_session(self._session)

        ##############################################################################
        #                                 Q-network                                  #
        ##############################################################################
        self._sensor_input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=(None, *self._sensor_state_shape),
            name="sensor_input",
        )
        self._heat_input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=(None, *self._heat_state_shape),
            name="heat_input"
        )
        # self._position_input_tensor = tf.placeholder(
        #     dtype=tf.float32,
        #     shape=(None, *self._position_state_shape),
        #     name="position_input"
        # )

        # conv1 = Conv2D(
        #     filters=self._n_sensors // 2,
        #     kernel_size=(1, self._window_size),
        #     data_format="channels_last",
        #     activation="relu",
        #     name="conv1",
        # )
        # conv_output = conv1(self._sensor_input_tensor)
        #
        # conv2 = Conv2D(
        #     filters=self._n_sensors // 4,
        #     kernel_size=(1, self._window_size),
        #     data_format="channels_last",
        #     activation="relu",
        #     name="conv2",
        # )
        # conv_output = conv2(conv_output)
        #
        # flatten_conv = Flatten(name="flatten_conv")
        # flattened_conv = flatten_conv(conv_output)

        concat = Concatenate(name="concat")
        concatenated_input = concat([
            self._sensor_input_tensor,
            self._heat_input_tensor,
            # self._position_input_tensor,
        ])

        hidden_dense0 = Dense(units=self._n_sensor_inputs * 8, activation="relu", name="hidden_dense1")
        x = hidden_dense0(concatenated_input)

        hidden_dense1 = Dense(units=self._n_sensor_inputs * 4, activation="relu", name="hidden_dense1")
        x = hidden_dense1(x)

        hidden_dense2 = Dense(units=self._n_sensor_inputs * 2, activation="relu", name="hidden_dense2")
        x = hidden_dense2(x)

        hidden_dense3 = Dense(units=self._n_sensor_inputs, activation="relu", name="hidden_dense3")
        x = hidden_dense3(x)

        output_layer = Dense(units=self._n_output_angles + 1, name="action_quality")
        self._actions_qualities_tensor = output_layer(x)
        self._action_index_tensor = tf.argmax(self._actions_qualities_tensor, axis=1, name="output")

        ################################################################################
        #                             Updating Q-network                               #
        ################################################################################
        self._chosen_actions_tensor = tf.placeholder(dtype=tf.int32, shape=(None,), name="chosen_actions")
        self._rewards_tensor = tf.placeholder(dtype=tf.float32, shape=(None,), name="discounted_rewards")
        self._terminates_tensor = tf.placeholder(dtype=tf.float32, shape=(None,), name="episode_terminated")
        self._replay_next_states_qualities_tensor = tf.placeholder(
            dtype=tf.float32, shape=self._actions_qualities_tensor.shape,
            name="replay_next_states_qualities"
        )

        next_state_indices = tf.stack((tf.range(0, tf.shape(self._rewards_tensor)[0]), self._chosen_actions_tensor),
                                      axis=1)
        responsible_qualities = tf.gather_nd(self._replay_next_states_qualities_tensor, next_state_indices)
        # noinspection PyTypeChecker
        target_quality = (
            self._rewards_tensor + self._terminates_tensor * responsible_qualities
            * self._process_config.reward_discount_coef
        )

        tf_range = tf.range(0, tf.shape(self._rewards_tensor)[0], dtype=tf.int32)
        state_indices = tf.stack((tf_range, self._chosen_actions_tensor), axis=1)
        current_quality = tf.gather_nd(self._actions_qualities_tensor, state_indices)

        loss = losses.mean_squared_error(
            target_quality,
            current_quality,
            reduction=losses.Reduction.MEAN
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        self._update_model = optimizer.minimize(loss)
        self._session.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

    def apply_action_from_network(self) -> None:
        action_index = self._session.run(self._action_index_tensor, feed_dict={
            self._sensor_input_tensor: self.get_sensor_input(),
            self._heat_input_tensor: self.get_heat_input(),
        })[0]
        self._apply_action(action_index)

    def apply_action_from_input(self,
                                prev_sensor_states: Deque[np.ndarray],
                                prev_heat_states: Deque[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:

        assert len(prev_sensor_states) == len(prev_heat_states) == self._n_steps_back - 1

        current_sensors = self.get_sensor_input()
        current_heat = self.get_heat_input()

        input_sensors = np.hstack((*prev_sensor_states, current_sensors)).reshape((1, *self._sensor_state_shape))
        input_heat = np.hstack((*prev_heat_states, current_heat)).reshape((1, *self._heat_state_shape))

        qualities = self._session.run(self._actions_qualities_tensor, feed_dict={
            self._sensor_input_tensor: input_sensors,
            self._heat_input_tensor: input_heat,
        })[0]
        print(qualities)
        action_index = np.argmax(qualities[:-1])  # type: int
        print(action_index)

        # action_index = self._session.run(self._action_index_tensor, feed_dict={
        #     self._sensor_input_tensor: input_sensors,
        #     self._heat_input_tensor: input_heat,
        # })[0]

        self._apply_action(action_index)

        return current_sensors, current_heat

    def load_model(self, path):
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(path)
        self._saver.restore(self._session, checkpoint.model_checkpoint_path)

    def train(self,
              save_path: Optional[str] = None,
              step_hook: Optional[Callable] = None) -> None:

        step_drop = (
            (self._process_config.start_random_action_prob - self._process_config.end_random_action_prob)
            / self._process_config.annealing_steps
        )
        random_action_prob = self._process_config.start_random_action_prob
        total_steps = 0
        pre_trained = False
        ep_buf = EpisodeBuffer(self._state_shapes, self._process_config.buffer_size)

        for i_episode in range(self._process_config.n_training_episodes):
            self._game_world.reset()
            current_sensor_state = np.tile(self.get_sensor_input(), self._n_steps_back)
            current_heat_state = np.tile(self.get_heat_input(), self._n_steps_back)
            # current_position_state = np.reshape(self._get_position_input(), (1, *self._position_state_shape))
            reward_sum = 0
            reward_sums = []

            for i_step in range(self._process_config.max_ep_length):

                total_steps += 1
                if not pre_trained and total_steps > self._process_config.pre_train_steps:
                    pre_trained = True
                    print("Pre-training ended.")

                if total_steps < self._process_config.pre_train_steps or np.random.rand(1) < random_action_prob:
                    action_index = np.array([np.random.randint(0, self._n_actions)])
                else:
                    action_index = self._session.run(self._action_index_tensor, feed_dict={
                        self._sensor_input_tensor: current_sensor_state.reshape(1, *self._sensor_state_shape),
                        self._heat_input_tensor: current_heat_state.reshape(1, *self._heat_state_shape),
                        # self._position_input_tensor: current_position_state,
                    })

                self._apply_action(action_index[0])

                reward = self._player.reward
                self._player.reset_reward_after_step()
                game_over = self._game_world.game_over
                step_state = self.get_sensor_input()
                next_sensor_state = np.hstack((current_sensor_state[self._n_sensors * self._n_sensor_types:],
                                               step_state))
                step_heat = self.get_heat_input()
                next_heat_state = np.hstack((current_heat_state[1:], step_heat))
                # next_position_state = np.reshape(self._get_position_input(), (1, *self._position_state_shape))

                ep_buf.add_experiences(Experiences(
                    [
                        current_sensor_state.reshape((1, *self._sensor_state_shape)),
                        current_heat_state.reshape((1, *self._heat_state_shape)),
                        # current_position_state
                    ],
                    [
                        next_sensor_state.reshape((1, *self._sensor_state_shape)),
                        next_heat_state.reshape((1, *self._heat_state_shape)),
                        # next_position_state
                    ],
                    action_index,
                    np.array([reward], dtype=np.float32),
                    np.array([game_over], dtype=np.bool),
                ))

                reward_sum += reward
                current_sensor_state[:] = next_sensor_state
                current_heat_state[:] = next_heat_state

                if total_steps > self._process_config.pre_train_steps:
                    if random_action_prob > self._process_config.end_random_action_prob:
                        random_action_prob -= step_drop

                    if total_steps % self._process_config.update_frequency == 0:
                        train_experiences = ep_buf.sample(self._process_config.replay_size)
                        (
                            sensors,
                            heat,
                            # position,
                        ) = train_experiences.states
                        (
                            next_sensors,
                            next_heat,
                            # next_position
                        ) = train_experiences.next_states
                        action_indices = train_experiences.action_indices
                        rewards = train_experiences.rewards
                        terminates = train_experiences.terminates

                        replay_next_state_qualities = self._session.run(self._actions_qualities_tensor, feed_dict={
                            self._sensor_input_tensor: next_sensors,
                            self._heat_input_tensor: next_heat,
                            # self._position_input_tensor: next_position,
                        })

                        self._session.run(self._update_model, feed_dict={
                            self._sensor_input_tensor: sensors,
                            self._heat_input_tensor: heat,
                            # self._position_input_tensor: position,
                            self._chosen_actions_tensor: action_indices,
                            self._rewards_tensor: rewards,
                            self._terminates_tensor: terminates,
                            self._replay_next_states_qualities_tensor: replay_next_state_qualities,
                        })

                if step_hook is not None:
                    step_hook()

                if game_over:
                    break

            reward_sums.append(reward_sum)

            if i_episode % 10 == 0:
                print(total_steps, np.mean(reward_sums[-10:]), random_action_prob)

            if save_path is not None and i_episode % 200 == 0:
                self._saver.save(self._session, "{save_path}/model-{i_episode}.ckpt".format(save_path=save_path,
                                                                                            i_episode=i_episode))
                print("Saved model at episode {i_episode}.".format(i_episode=i_episode))

        self._saver.save(self._session, "{save_path}/model-final.ckpt".format(save_path=save_path))
