from typing import Tuple, NamedTuple, Callable, Optional

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten
from numpy.core.multiarray import ndarray
from tensorflow import losses

from GameObjects import World, PlayerMovementDirection


class Experiences(NamedTuple):
    states: np.ndarray
    next_states: np.ndarray
    action_indices: np.ndarray
    rewards: np.ndarray
    terminates: np.ndarray


class EpisodeBuffer:
    _buffer_size: int

    _array_size: int
    _current_ptr: int
    _indices_for_sampling: np.ndarray

    _state_buffer: np.ndarray
    _next_state_buffer: ndarray
    _action_index_buffer: ndarray
    _reward_buffer: ndarray
    _terminate_buffer: ndarray

    def __init__(self, state_shape: Tuple, buffer_size: int = 50000):
        self._buffer_size = buffer_size

        self._array_size = self._buffer_size * 2
        self._current_ptr = 0
        self._indices_for_sampling = np.arange(0, self._buffer_size, 1)

        self._state_buffer = np.empty(
            shape=(self._array_size, *state_shape),
            dtype=np.float32
        )
        self._next_state_buffer = np.empty(
            shape=(self._array_size, *state_shape),
            dtype=np.float32
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

    def add_experiences(self, experiences: Experiences) -> None:
        states, next_states, action_indices, rewards, terminates = experiences
        assert (
            states.shape[0] == action_indices.shape[0] == rewards.shape[0] == next_states.shape[0] ==
            terminates.shape[0]
        )
        n_experiences = states.shape[0]

        if self._current_ptr - 1 + n_experiences >= self._array_size:
            self._state_buffer[0:self._buffer_size] = \
                self._state_buffer[self._current_ptr - self._buffer_size: self._current_ptr]
            self._next_state_buffer[0:self._buffer_size] = \
                self._next_state_buffer[self._current_ptr - self._buffer_size: self._current_ptr]
            self._action_index_buffer[0:self._buffer_size] = \
                self._action_index_buffer[self._current_ptr - self._buffer_size: self._current_ptr]
            self._reward_buffer[0:self._buffer_size] = \
                self._reward_buffer[self._current_ptr - self._buffer_size: self._current_ptr]
            self._terminate_buffer[0:self._buffer_size] = \
                self._terminate_buffer[self._current_ptr - self._buffer_size: self._current_ptr]
            self._current_ptr = 0

        self._state_buffer[self._current_ptr: self._current_ptr + n_experiences] = states
        self._next_state_buffer[self._current_ptr:  self._current_ptr + n_experiences] = next_states
        self._action_index_buffer[self._current_ptr: self._current_ptr + n_experiences] = action_indices
        self._reward_buffer[self._current_ptr: self._current_ptr + n_experiences] = rewards
        self._terminate_buffer[self._current_ptr: self._current_ptr + n_experiences] = terminates
        self._current_ptr += n_experiences

    def sample(self, sample_size: int) -> Experiences:
        indices = np.random.choice(
            self._current_ptr - self._buffer_size + self._indices_for_sampling,
            size=sample_size,
            replace=False
        )
        return Experiences(
            self._state_buffer[indices],
            self._next_state_buffer[indices],
            self._action_index_buffer[indices],
            self._reward_buffer[indices],
            self._terminate_buffer[indices]
        )


class LearningProcessConfig(NamedTuple):
    replay_size: int
    update_frequency: int
    reward_discount_coef: float
    start_random_action_prob: float
    end_random_action_prob: float
    annealing_steps: int
    n_training_episodes: int
    pre_train_steps: int
    max_ep_length: int


class Learner:
    _game_world: World
    _n_output_angles: int
    _session: tf.Session
    _window_size: int
    _time_between_actions_s: float
    _process_config: LearningProcessConfig

    _n_extra_inputs: int
    _n_sensor_inputs: int

    _output_angles: np.ndarray
    _sensor_input_array: np.ndarray
    _n_sensors: int
    _sensor_input_tensor: tf.Tensor
    _heat_input_tensor: tf.Tensor
    _action_index_tensor: tf.Tensor
    _chosen_actions_tensor: tf.Tensor
    _rewards_tensor: tf.Tensor
    _terminates_tensor: tf.Tensor
    _actions_qualities: tf.Tensor
    _replay_next_states_qualities_tensor: tf.Tensor
    _replay_states_qualities: tf.Tensor
    _update_model: tf.Operation

    def __init__(self,
                 world: World,
                 n_output_angles: int,
                 session: tf.Session,
                 window_size: int,
                 time_between_actions_s: float,
                 process_config: LearningProcessConfig):
        assert window_size % 2 == 1

        self._game_world = world
        self._window_size = window_size
        self._n_output_angles = n_output_angles
        self._session = session
        self._time_between_actions_s = time_between_actions_s
        self._process_config = process_config

        self._n_actions = self._n_output_angles + 1
        self._output_angles = np.linspace(-np.pi, np.pi, self._n_output_angles, False)
        self._n_sensors = self._game_world.proximity_sensors_np.n_sensors
        self._n_extra_inputs = self._window_size // 2
        self._n_sensor_inputs = self._n_sensors + self._n_extra_inputs * 2

        self._state_shape = (1, self._n_sensor_inputs, 5)
        self._episode_buffer = EpisodeBuffer(self._state_shape, 5)

    def _get_sensor_input(self) -> np.ndarray:
        distances = self._game_world.proximity_sensors_np.distances
        object_types = self._game_world.proximity_sensors_np.object_types
        prox_sens_data = np.zeros(shape=(1, self._n_sensors, 5), dtype=np.float32)
        prox_sens_data[0, :, 0] = distances
        prox_sens_data[0, np.arange(0, self._n_sensors), object_types] = 1

        result_indices = np.fmod(
            np.arange(0, self._n_sensors + self._n_extra_inputs * 2) - self._n_extra_inputs,
            self._n_sensors
        )
        return prox_sens_data[0, result_indices, :]

    def initialize(self) -> None:
        keras.backend.set_session(self._session)

        ##############################################################################
        #                                 Q-network                                  #
        ##############################################################################
        sensor_input_shape = (None, 1, self._n_sensor_inputs, 5)
        self._sensor_input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=sensor_input_shape,
            name="sensor_input",
        )
        self._heat_input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="heat_input")

        conv1 = Conv2D(
            filters=self._n_sensors // 4,
            kernel_size=(1, self._window_size),
            data_format="channels_last",
            activation="relu",
            name="conv1",
        )
        x = conv1(self._sensor_input_tensor)

        conv2 = Conv2D(
            filters=self._n_sensors // 4,
            kernel_size=(1, self._window_size),
            data_format="channels_last",
            activation="relu",
            name="conv2",
        )
        x = conv2(x)

        flatten_conv = Flatten(name="flatten_conv")
        x = flatten_conv(x)

        hidden_dense1 = Dense(units=self._n_sensors, activation="relu", name="hidden_dense1")
        x = hidden_dense1(x)

        hidden_dense2 = Dense(units=self._n_sensors, activation="relu", name="hidden_dense2")
        x = hidden_dense2(x)

        output_layer = Dense(units=self._n_output_angles + 1, name="action_quality", )
        self._actions_qualities = output_layer(x)
        self._action_index_tensor = tf.argmax(self._actions_qualities, axis=1, name="output")

        ################################################################################
        #                             Updating Q-network                               #
        ################################################################################
        self._chosen_actions_tensor = tf.placeholder(dtype=np.int32, shape=(None,), name="chosen_actions")
        self._rewards_tensor = tf.placeholder(dtype=tf.float32, shape=(None,), name="discounted_rewards")
        self._terminates_tensor = tf.placeholder(dtype=tf.float32, shape=(None,), name="episode_terminated")
        self._replay_next_states_qualities_tensor = tf.placeholder(
            dtype=tf.float32, shape=self._actions_qualities.shape,
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

        tf_range = tf.range(0, tf.shape(self._rewards_tensor)[0], dtype=np.int32)
        state_indices = tf.stack((tf_range, self._chosen_actions_tensor), axis=1)
        current_quality = tf.gather_nd(self._actions_qualities, state_indices)

        loss = losses.mean_squared_error(
            target_quality,
            current_quality,
            reduction=losses.Reduction.MEAN
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        self._update_model = optimizer.minimize(loss)
        self._session.run(tf.global_variables_initializer())

    def _get_angle_movement(self, action_index: int) -> Tuple[float, PlayerMovementDirection]:
        if action_index == self._n_output_angles:
            return self._game_world.player.angle, PlayerMovementDirection.NONE
        else:
            return self._output_angles[action_index], PlayerMovementDirection.FORWARD

    def get_next_action(self) -> Tuple[float, PlayerMovementDirection]:
        sensor_input = np.reshape(self._get_sensor_input(), (1, 1, self._n_sensor_inputs, 5))
        feed_dict = {
            self._sensor_input_tensor: sensor_input,
            self._heat_input_tensor: np.array([[self._game_world.player.heat]]),
        }

        action_index = self._session.run(self._action_index_tensor, feed_dict=feed_dict)[0]

        return self._get_angle_movement(action_index)

    def apply_action(self, elapsed_time_s: float) -> None:
        sensor_input = np.reshape(self._get_sensor_input(), (1, 1, self._n_sensor_inputs, 5))
        feed_dict = {
            self._sensor_input_tensor: sensor_input,
            self._heat_input_tensor: np.array([[self._game_world.player.heat]]),
        }

        action_index = self._session.run(self._action_index_tensor, feed_dict=feed_dict)[0]

        angle, movement = self._get_angle_movement(action_index)

        self._game_world.update_player_angle(angle)
        self._game_world.move_player(elapsed_time_s, movement)
        self._game_world.update_state(elapsed_time_s)

        reward = self._game_world.player.reward_sum
        game_over = self._game_world.game_over
        next_sensor_input = np.reshape(self._get_sensor_input(), (1, 1, self._n_sensor_inputs, 5))

        self._episode_buffer.add_experiences(Experiences(
            sensor_input,
            next_sensor_input,
            np.array([action_index], dtype=np.int32),
            np.array([reward], dtype=np.float32),
            np.array([game_over], dtype=np.bool),
        ))

    def train(self, save_path: Optional[str] = None, step_hook: Optional[Callable] = None) -> None:
        saver = tf.train.Saver()

        step_drop = (
            (self._process_config.start_random_action_prob - self._process_config.end_random_action_prob)
            / self._process_config.annealing_steps
        )
        total_steps = 0
        pre_trained = False

        for i_episode in range(self._process_config.n_training_episodes):
            ep_buf = EpisodeBuffer(self._state_shape, 1000)
            self._game_world.reset()
            random_action_prob = self._process_config.start_random_action_prob
            current_state = np.reshape(self._get_sensor_input(), (1, 1, self._n_sensor_inputs, 5))
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
                        self._sensor_input_tensor: current_state,
                        self._heat_input_tensor: np.array([[0]], dtype=np.float32)
                    })

                angle, movement = self._get_angle_movement(action_index[0])

                self._game_world.update_player_angle(angle)
                self._game_world.move_player(self._time_between_actions_s, movement)
                self._game_world.update_state(self._time_between_actions_s)

                reward = self._game_world.player.reward_sum
                game_over = self._game_world.game_over
                next_state = np.reshape(self._get_sensor_input(), (1, 1, self._n_sensor_inputs, 5))

                ep_buf.add_experiences(Experiences(
                    current_state,
                    next_state,
                    action_index,
                    np.array([reward], dtype=np.float32),
                    np.array([game_over], dtype=np.bool),
                ))

                reward_sum += reward
                current_state = next_state

                if total_steps > self._process_config.pre_train_steps:
                    if random_action_prob > self._process_config.end_random_action_prob:
                        random_action_prob -= step_drop

                    if i_step % self._process_config.update_frequency == 0:
                        train_experiences = ep_buf.sample(self._process_config.replay_size)
                        states = train_experiences.states
                        next_states = train_experiences.next_states
                        action_indices = train_experiences.action_indices
                        rewards = train_experiences.rewards
                        terminates = train_experiences.terminates

                        replay_next_state_qualities = self._session.run(self._actions_qualities, feed_dict={
                            self._sensor_input_tensor: next_states,
                            self._heat_input_tensor: np.zeros(shape=(next_states.shape[0], 1), dtype=np.float32)
                        })

                        self._session.run(self._update_model, feed_dict={
                            self._sensor_input_tensor: states,
                            self._heat_input_tensor: np.zeros(shape=(states.shape[0], 1), dtype=np.float32),
                            self._chosen_actions_tensor: action_indices,
                            self._rewards_tensor: rewards,
                            self._terminates_tensor: terminates,
                            self._replay_next_states_qualities_tensor: replay_next_state_qualities,
                        })

                if step_hook is not None:
                    step_hook()

                if game_over:
                    break

            print("finished an episode")

            reward_sums.append(reward_sum)

            if i_episode % 10 == 0:
                print(total_steps, np.mean(reward_sums[-10:]), random_action_prob)

            if save_path is not None and i_episode % 100 == 0:
                saver.save(self._session, f"{save_path}/model-{i_episode}.ckpt")
                print(f"Saved model at episode {i_episode}.")
