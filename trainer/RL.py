from collections import namedtuple
from pprint import pprint
from typing import Tuple, Callable, Optional, List, Type, Dict, Any, Union

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Concatenate, Conv2D, LSTM, Layer
from keras.layers.advanced_activations import PReLU
from tensorflow import losses
from tensorflow.python.training.saver import Saver

from trainer.GameObjects import World, PlayerMovementDirection, Player, SensedObject

Experiences = namedtuple("Experiences", [
    "states",
    "next_states",
    "actions",
    "rewards",
    "terminates"
])


# Experience = namedtuple("Experience", [
#     "state",
#     "next_state",
#     "action_index",
#     "reward",
#     "terminated",
# ])
#

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

    def __init__(self, state_shapes: List[Tuple], buffer_size: int, action_dtype: Type[np.int32]):
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
            dtype=action_dtype
        )
        self._reward_buffer = np.empty(
            shape=(self._array_size,),
            dtype=np.float32,
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
        self._fill_buffer(self._reward_buffer, rewards, n_experiences)
        self._fill_buffer(self._terminate_buffer, terminates, n_experiences)

        self._current_ptr += n_experiences

    # def get_indices(self, sample_size: int):
    #     if self._current_ptr < self._buffer_size:
    #         # print("first")
    #         indices = np.random.choice(
    #             self._current_ptr,
    #             size=sample_size,
    #             replace=False
    #         )
    #     else:
    #         # print("second")
    #         indices = np.random.choice(
    #             self._current_ptr - self._buffer_size + self._indices_for_sampling,
    #             size=sample_size,
    #             replace=False
    #         )
    #     return indices

    def sample(self, sample_size: int) -> Experiences:
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
#
#     def sample_indices(self, indices: np.ndarray) -> Experiences:
#         return Experiences(
#             [buf[indices] for buf in self._state_buffers],
#             [buf[indices] for buf in self._next_states_buffer],
#             self._action_index_buffer[indices],
#             self._reward_buffer[indices],
#             self._terminate_buffer[indices]
#         )
#
#
# class EpisodeBufferSimple:
#     _buffer_size = ...  # type: int
#     _buffer = ...  # type: Deque[Experience]
#
#     def __init__(self, buffer_size: int):
#         self._buffer_size = buffer_size
#         self._buffer = deque(maxlen=self._buffer_size)
#
#     def add_experience(self, experience: Experience) -> None:
#         self._buffer.append(experience)
#
#     def sample(self, sample_size: int) -> List[Experience]:
#         indices = np.random.choice(
#             len(self._buffer),
#             size=sample_size,
#             replace=False,
#         )
#         return [self._buffer[index] for index in indices]
#
#     def sample_indices(self, indices: np.ndarray) -> List[Experience]:
#         return [self._buffer[index] for index in indices]
#
#
# def experience_list_to_experiences(state_shapes: List[Tuple], experience_list: List[Experience]) -> Experiences:
#     n_experiences = len(experience_list)
#     states = [np.zeros((n_experiences, *state_shape), dtype=np.float32) for state_shape in state_shapes]
#     next_states = [np.zeros((n_experiences, *state_shape), dtype=np.float32) for state_shape in state_shapes]
#     action_indices = np.zeros(n_experiences, dtype=np.int32)
#     rewards = np.zeros(n_experiences, dtype=np.float32)
#     terminates = np.zeros(n_experiences, dtype=np.bool)
#
#     for i, experience in enumerate(experience_list):
#         state, next_state, action_index, reward, terminated = experience
#         assert len(state_shapes) == len(state) == len(next_state)
#         for j, (state_aspect, next_state_aspect) in enumerate(zip(state, next_state)):
#             states[j][i, :] = state_aspect
#             next_states[j][i, :] = next_state_aspect
#         action_indices[i] = action_index
#         rewards[i] = reward
#         terminates[i] = terminated
#
#     return Experiences(
#         states,
#         next_states,
#         action_indices,
#         rewards,
#         terminates,
#     )
#
#
# def assert_experiences_equals(e1: Experiences, e2: Experiences):
#     assert len(e1.states) == len(e2.states) == len(e1.next_states) == len(e2.next_states)
#     # print(len(e1.states))
#     for i in range(len(e1.states)):
#         assert e1.states[i].shape == e2.states[i].shape == e1.next_states[i].shape == e2.next_states[i].shape
#         # print(i)
#         # print(e1.states[i].shape)
#         # print(e2.states[i].shape)
#         # print(e1.states[i])
#         # print(e2.states[i])
#         assert np.allclose(e1.states[i], e2.states[i])
#         assert np.allclose(e1.next_states[i], e2.next_states[i])
#     assert np.allclose(e1.action_indices, e2.action_indices)
#     # print(e1.rewards, e2.rewards)
#     assert np.allclose(e1.rewards, e2.rewards)
#     assert np.allclose(e1.terminates, e2.terminates)


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
    "n_skipped_frames",
    "target_network_update_frequency",
    "initial_temperature",
    "temp_coef",
    "min_temperature",
    "framerate",
    "regularization_loss_coef",
    "learning_rate",
])


class BaseConfig:
    activation = ...  # type: str
    name = ...  # type: Optional[str]

    def __init__(self, activation: str, name: Optional[str]):
        self.activation = activation.lower()
        self.name = name

        assert self.activation in ["linear", "relu", "tanh", "hard_sigmoid", "prelu"]


class ConvConfig(BaseConfig):
    filters = ...  # type: int

    def __init__(self, filters: int, activation: Optional[str] = None, name: Optional[str] = None):
        super().__init__(activation if activation is not None else "linear", name)
        self.filters = filters


class LSTMConfig(BaseConfig):
    units = ...  # type: int
    recurrent_activation = ...  # type: str

    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 recurrent_activation: Optional[str] = None,
                 name: Optional[str] = None):
        super().__init__(activation if activation is not None else "tanh", name)
        self.units = units
        self.recurrent_activation = recurrent_activation if recurrent_activation is not None else "hard_sigmoid"


class DenseConfig(BaseConfig):
    units = ...  # type: int

    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 name: Optional[str] = None):
        super().__init__(activation if activation is not None else "linear", name)
        self.units = units


def config_to_dict(config: BaseConfig) -> Dict[str, Any]:
    return vars(config)


def dict_to_config(inp_dict: Dict[str, Any], t_config: Union[Type[ConvConfig], Type[LSTMConfig], Type[DenseConfig]]):
    return t_config(**inp_dict)


def config_to_layers(config: BaseConfig,
                     additional_kwargs: Dict[str, Any],
                     t_layer: Type[Layer]) -> List[Layer]:
    kwargs = dict(vars(config).items())
    kwargs.update(additional_kwargs)
    if kwargs["activation"] == "prelu":
        use_prelu = True
        kwargs["activation"] = "linear"
    else:
        use_prelu = False

    curr_layer = t_layer(**kwargs)
    result = [curr_layer]
    if use_prelu:
        result.append(PReLU())

    return result


class NetworkConfig:
    window_size = ...  # type: int
    conv_configs = ...  # type: List[ConvConfig]
    lstm_configs = ...  # type: List[LSTMConfig]
    dense_configs = ...  # type: List[DenseConfig]
    n_output_angles = ...  # type: int

    def __init__(self,
                 window_size: int,
                 n_output_angles: int,
                 conv_configs: List[ConvConfig],
                 lstm_configs: List[LSTMConfig],
                 dense_configs: List[DenseConfig]):
        self.window_size = window_size
        self.conv_configs = conv_configs
        self.lstm_configs = lstm_configs
        self.dense_configs = dense_configs
        self.n_output_angles = n_output_angles


def network_config_to_dict(network_config: NetworkConfig):
    return {
        "window_size": network_config.window_size,
        "conv_configs": [config_to_dict(conv_config) for conv_config in network_config.conv_configs],
        "lstm_configs": [config_to_dict(lstm_config) for lstm_config in network_config.lstm_configs],
        "dense_configs": [config_to_dict(dense_config) for dense_config in network_config.dense_configs],
        "n_output_angles": network_config.n_output_angles,
    }


def dict_to_network_config(inp_dict: Dict[str, Any]) -> NetworkConfig:
    return NetworkConfig(
        window_size=inp_dict["window_size"],
        n_output_angles=inp_dict["n_output_angles"],
        conv_configs=[dict_to_config(dict_conv_config, ConvConfig) for dict_conv_config in inp_dict["conv_configs"]],
        lstm_configs=[dict_to_config(dict_lstm_config, LSTMConfig) for dict_lstm_config in inp_dict["lstm_configs"]],
        dense_configs=[dict_to_config(dict_dense_config, DenseConfig) for dict_dense_config in
                       inp_dict["dense_configs"]],
    )


class ActorCriticRecurrentLearner:
    _game_world = ...  # type: World
    _session = ...  # type: tf.Session
    _network_config = ...  # type: NetworkConfig
    _process_config = ...  # type: LearningProcessConfig

    _time_between_actions_s = ...  # type: float
    _player = ...  # type: Player
    _n_sensor_types = ...  # type: int
    _n_actions = ...  # type: int
    _n_sensors = ...  # type: int
    _n_sensor_inputs = ...  # type: int
    _max_sensor_distance = ...  # type: float
    _sensor_state_shape = ...  # type: Tuple
    _heat_state_shape = ...  # type: Tuple
    _exploration_pressure_shape = ...  # type: Tuple
    _previous_reward_shape = ...  # type: Tuple
    _previous_action_shape = ...  # type: Tuple
    _exploration_temp_state_shape = ...  # type: Tuple
    _state_shapes = ...  # type: List[Tuple]
    _state_names = ...  # type: List[str]

    _decision_input_tensors = ...  # type: List[tf.Tensor]
    _decision_layers = ...  # type: List[Layer]
    _decision_output = ...  # type: tf.Tensor
    _decision_value = ...  # type: tf.Tensor
    _train_input_tensors = ...  # type: List[tf.Tensor]
    _train_layers = ...  # type: List[Layer]
    _train_output = ...  # type: tf.Tensor
    _train_value_output = ...  # type: tf.Tensor

    _previous_heat = ...  # type: float
    _previous_exploration_pressure = ...  # type: float
    _previous_reward = ...  # type: float
    _previous_action = ...  # type: int

    _update_chosen_actions = ...  # type: tf.Tensor
    _update_true_cumul_rewards = ...  # type: tf.Tensor
    _update_values_tensor = ...  # type: tf.Tensor
    _update_op = ...  # type: tf.Operation
    _saver = ...  # type: Saver

    def __init__(self,
                 world: World,
                 session: tf.Session,
                 network_config: NetworkConfig,
                 process_config: LearningProcessConfig):
        self._game_world = world
        self._session = session
        self._network_config = network_config
        self._process_config = process_config

        self._time_between_actions_s = self._process_config.framerate
        self._player = self._game_world.player
        # noinspection PyTypeChecker
        self._n_sensor_types = len(SensedObject) + 1
        self._n_sensors = self._game_world.proximity_sensors_np.n_sensors
        self._n_extra_inputs = self._network_config.window_size // 2
        self._n_sensor_inputs = self._n_sensors + self._n_extra_inputs * 2
        self._max_sensor_distance = self._game_world.proximity_sensors_np.max_distance

        self._sensor_state_shape = (1, self._n_sensor_inputs, self._n_sensor_types)
        self._heat_state_shape = (1,)
        self._exploration_pressure_shape = (1,)
        self._previous_reward_shape = (1,)
        self._previous_action_shape = (self._network_config.n_output_angles,)
        self._exploration_temp_state_shape = (1,)

        self._state_shapes = [
            self._sensor_state_shape,
            self._heat_state_shape,
            self._exploration_pressure_shape,
            self._previous_reward_shape,
            self._previous_action_shape,
            self._exploration_temp_state_shape,
        ]

        self._state_names = [
            "sensor",
            "heat",
            "exploration_pressure",
            "previous_reward",
            "previous_action",
            "exploration_temp"
        ]

        self._sensor_input_index = 0
        self._exploration_temp_index = len(self._state_names) - 1

        self._previous_heat = 0.
        self._previous_exploration_pressure = 0.
        self._previous_reward = 0.
        self._previous_action = 0
        self._current_exploration_temperature = float(self._process_config.initial_temperature)

    def _get_sensor_input(self) -> np.ndarray:
        distances = self._game_world.proximity_sensors_np.distances
        object_types = self._game_world.proximity_sensors_np.object_types
        prox_sens_data = np.zeros(shape=(1, self._n_sensors, self._n_sensor_types), dtype=np.float32)
        prox_sens_data[0, :, 0] = distances / self._max_sensor_distance
        prox_sens_data[0, np.arange(self._n_sensors), object_types] = 1

        result_indices = np.fmod(
            np.arange(0, self._n_sensors + self._n_extra_inputs * 2) - self._n_extra_inputs,
            self._n_sensors
        )
        return np.reshape(prox_sens_data[0, result_indices, :], (1, *self._sensor_state_shape))

    def _get_heat_input(self) -> np.ndarray:
        res = float(self._previous_heat)
        self._previous_heat = None
        return np.array([[res]], dtype=np.float32)

    def _get_exploration_pressure_input(self) -> np.ndarray:
        res = float(self._previous_exploration_pressure)
        self._previous_exploration_pressure = None
        return np.array([[res]], dtype=np.float32)

    def _get_previous_reward(self) -> np.ndarray:
        res = float(self._previous_reward)
        self._previous_reward = None
        return np.array([[res]], dtype=np.float32)

    def _get_previous_action(self) -> np.ndarray:
        result = np.zeros(shape=(1, self._network_config.n_output_angles), dtype=np.float32)
        result[0, self._previous_action] = 1
        self._previous_action = None
        return result

    def _get_exploration_temp(self) -> np.ndarray:
        return np.array([[float(self._current_exploration_temperature)]], dtype=np.float32)

    def _get_input_list(self) -> List[np.ndarray]:
        return [
            self._get_sensor_input(),
            self._get_heat_input(),
            self._get_exploration_pressure_input(),
            self._get_previous_reward(),
            self._get_previous_action(),
            self._get_exploration_temp(),
        ]

    def _initialize_network(self, decision_mode: bool) -> Tuple[List[tf.Tensor], List[Layer], tf.Tensor, tf.Tensor]:
        scope_name = "decision" if decision_mode else "train"
        with tf.variable_scope(scope_name):

            batch_size = 1 if decision_mode else None
            input_tensors = [
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(batch_size, *shape),
                    name="{}_input".format(name)
                )
                for shape, name in zip(self._state_shapes, self._state_names)
            ]

            conved_input = input_tensors[self._sensor_input_index]
            trainable = not decision_mode
            layers = []

            x = conved_input
            for conv_config in self._network_config.conv_configs:
                additional_kwargs = {
                    "kernel_size": (1, self._network_config.window_size),
                    "data_format": "channels_last",
                    "trainable": trainable,
                }

                local_layers = config_to_layers(conv_config, additional_kwargs, Conv2D)
                for local_layer in local_layers:
                    x = local_layer(x)
                    layers.append(local_layer)

            _, _, n_pixels, n_filters = x.shape
            flattened_conv = tf.reshape(
                tensor=x,
                shape=(-1, n_pixels * n_filters),
                name="flattened_conv",
            )

            concatted = Concatenate(name="concat")(
                [flattened_conv]
                + [
                    input_tensor
                    for i, input_tensor in enumerate(input_tensors)
                    if i != self._sensor_input_index and i != self._exploration_temp_index
                ]
            )

            _, n_units = concatted.shape
            n_units = int(n_units)

            if decision_mode:
                x = tf.reshape(concatted, (1, 1, n_units), name="pre_lstm_reshaped")
                stateful = True
                return_sequences_after_last = False
            else:
                x = tf.reshape(concatted, (1, -1, n_units), name="pre_lstm_reshaped")
                stateful = False
                return_sequences_after_last = True

            for i, lstm_config in enumerate(self._network_config.lstm_configs):
                return_sequences = (
                    True if i < len(self._network_config.lstm_configs) - 1 else return_sequences_after_last
                )

                additional_kwargs = {
                    "stateful": stateful,
                    "return_sequences": return_sequences,
                    "trainable": trainable,
                }

                local_layers = config_to_layers(lstm_config, additional_kwargs, LSTM)
                for local_layer in local_layers:
                    x = local_layer(x)
                    layers.append(local_layer)

            if not decision_mode:
                x = tf.reshape(x, (-1, x.shape[2]))

            for dense_config in self._network_config.dense_configs:
                additional_kwargs = {
                    "trainable": trainable,
                }
                local_layers = config_to_layers(dense_config, additional_kwargs, Dense)
                for local_layer in local_layers:
                    x = local_layer(x)
                    layers.append(local_layer)

            output_layer = Dense(
                units=self._network_config.n_output_angles,
                activation="linear",
                name="output_layer",
                trainable=trainable,
            )
            output_angle_values = output_layer(x)
            layers.append(output_layer)

            current_temp = input_tensors[self._exploration_temp_index]
            cooled = output_angle_values / current_temp
            output_angle_probabilities = tf.nn.softmax(cooled, name="output_angle_probabilites")

            value_output_layer = Dense(
                units=1,
                name="value_layer",
                trainable=trainable,
            )
            output_value = value_output_layer(x)
            layers.append(value_output_layer)

            return input_tensors, layers, output_angle_probabilities, output_value

    def _copy_weights_from_train_to_decision(self) -> None:
        for decision_layer, train_layer in zip(self._decision_layers, self._train_layers):
            decision_layer.set_weights(train_layer.get_weights())

    def _get_angle(self, action_index: int) -> float:
        return self._game_world.player.angle + np.linspace(
            -np.pi, np.pi, self._network_config.n_output_angles, False
        )[action_index]

    def initialize_a2c(self) -> None:
        tf.logging.info("Initializing A2C")

        keras.backend.set_session(self._session)

        (
            self._decision_input_tensors,
            self._decision_layers,
            self._decision_output,
            self._decision_value,
        ) = self._initialize_network(True)
        (
            self._train_input_tensors,
            self._train_layers,
            self._train_output,
            self._train_value_output
        ) = self._initialize_network(False)

        self._update_chosen_actions = tf.placeholder(
            dtype=tf.int32, shape=(None,), name="update_chosen_actions"
        )
        self._update_true_cumul_rewards = tf.placeholder(
            dtype=tf.float32, shape=(None,), name="update_true_cumul_rewards"
        )
        self._update_values_tensor = tf.placeholder(dtype=tf.float32, shape=(None,), name="update_values")

        tf_range = tf.range(0, tf.shape(self._update_chosen_actions)[0], dtype=tf.int32)
        state_indices = tf.stack((tf_range, self._update_chosen_actions), axis=1)
        chosen_probs = tf.gather_nd(self._train_output, state_indices)

        advantages = self._update_true_cumul_rewards - self._update_values_tensor

        n = tf.cast(tf.shape(self._train_output)[0], dtype=tf.float32)
        self._policy_loss = policy_loss = -1 / n * tf.reduce_sum(advantages * tf.log(chosen_probs))

        self._value_loss = value_loss = (
                    1 / n * tf.reduce_sum(tf.square(self._update_true_cumul_rewards - self._train_value_output)))

        entropies = -tf.reduce_sum(self._train_output * tf.log(self._train_output), axis=1)
        self._regularization_loss = regularization_loss = -1 / n * tf.reduce_sum(entropies)

        all_loss = policy_loss + value_loss + self._process_config.regularization_loss_coef * regularization_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self._process_config.learning_rate)

        self._vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        self._gradients = tf.gradients(all_loss, self._vars)

        self._update_op = optimizer.apply_gradients(zip(self._gradients, self._vars))

        self._session.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver()

    def load_model(self, path: str, step: int):
        tf.logging.info("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(path)
        checkpoint_end = "model-{}.ckpt".format(step)

        for checkpoint_path in checkpoint.all_model_checkpoint_paths:
            if checkpoint_path.endswith(checkpoint_end):
                our_checkpoint_path = checkpoint_path
                break
        else:
            raise ValueError("wtf")

        self._saver.restore(self._session, our_checkpoint_path)

    def _update_previous(self, angle_index: int, step_hook: Optional[Callable[[], None]]):
        new_angle = self._get_angle(angle_index)
        self._previous_action = angle_index
        self._previous_reward = 0.
        self._previous_heat = 0.
        self._previous_exploration_pressure = 0.
        for _ in range(self._process_config.n_skipped_frames):
            self._previous_reward += self._game_world.update_world_and_player_and_get_reward(
                self._time_between_actions_s, new_angle, PlayerMovementDirection.FORWARD
            )
            self._previous_heat += self._game_world.player.heat
            self._previous_exploration_pressure += self._game_world.exploration_pressure
            if step_hook is not None:
                step_hook()
            if self._game_world.game_over:
                break

    def _update_temperature(self) -> None:
        if self._current_exploration_temperature <= self._process_config.min_temperature:
            return
        self._current_exploration_temperature -= self._process_config.temp_coef * self._current_exploration_temperature

    def loop(self, temperature: float, step_hook: Optional[Callable[[], None]] = None):
        # print("Start the loop!")

        self._current_exploration_temperature = temperature

        i = 0
        while True:
            inputs = self._get_input_list()

            decision_output = self._session.run(
                self._decision_output,
                feed_dict=dict(list(zip(self._decision_input_tensors, inputs)))
            )
            assert not np.any(np.isnan(decision_output))
            new_angle_index = np.random.choice(np.arange(self._network_config.n_output_angles), p=decision_output[0])

            self._update_previous(new_angle_index, step_hook)

            # i += self._process_config.n_skipped_frames
            # if i > self._process_config.max_ep_length:
            #     break

    def train(self, save_path: Optional[str] = None, step_hook: Optional[Callable[[], None]] = None) -> None:
        tf.logging.info("Starting training A2C agent")

        total_steps = 0
        total_updates = 0
        t_max = self._process_config.update_frequency
        gamma = self._process_config.reward_discount_coef
        self._current_exploration_temperature = float(self._process_config.initial_temperature)

        reward_sums = []

        for i_episode in range(self._process_config.n_training_episodes):
            self._game_world.reset()

            exps = []
            i_step = 0

            while True:
                total_steps += 1
                i_step += 1

                inputs = self._get_input_list()

                value, decision_probs = self._session.run(
                    [self._decision_value, self._decision_output],
                    feed_dict=dict(list(zip(self._decision_input_tensors, inputs)))
                )
                assert not np.any(np.isnan(decision_probs))
                new_angle_index = np.random.choice(np.arange(self._network_config.n_output_angles), p=decision_probs[0])

                self._update_previous(new_angle_index, step_hook)
                self._update_temperature()

                reward_sums.append(self._previous_reward)

                exps.append(Exp(new_angle_index, value, self._previous_reward, inputs))

                game_over = self._game_world.game_over

                if game_over or len(exps) >= t_max:
                    total_updates += 1

                    cumul_rewards = np.empty(shape=(len(exps),), dtype=np.float32)
                    chosen_actions = np.empty(shape=(len(exps),), dtype=np.int32)
                    values = np.empty(shape=(len(exps),), dtype=np.float32)
                    mem_inputs = [
                        np.empty(shape=(len(exps), *state_shape), dtype=np.float32)
                        for state_shape in self._state_shapes
                    ]

                    all_rewards = np.array([exp.reward for exp in exps], dtype=np.float32)
                    std_rewards = np.std(all_rewards)
                    norm_rewards = (all_rewards - np.mean(all_rewards)) / (std_rewards if std_rewards != 0 else 1)

                    current_cumul_reward = 0 if game_over else value
                    for i_exp in reversed(range(len(exps))):
                        cumul_rewards[i_exp] = norm_rewards[i_exp] + gamma * current_cumul_reward
                        current_cumul_reward = cumul_rewards[i_exp]

                        chosen_actions[i_exp] = exps[i_exp].chosen_action
                        values[i_exp] = exps[i_exp].value
                        for i_state, input_state in enumerate(exps[i_exp].inputs):
                            mem_inputs[i_state][i_exp] = input_state

                    exps.clear()

                    fd = dict(list(zip(self._train_input_tensors, mem_inputs)))
                    fd[self._update_chosen_actions] = chosen_actions
                    fd[self._update_values_tensor] = values
                    fd[self._update_true_cumul_rewards] = cumul_rewards

                    policy_loss, value_loss, reg_loss = self._session.run(
                        [self._policy_loss, self._value_loss, self._regularization_loss, self._update_op],
                        fd
                    )

                    tf.logging.debug(
                        "policy loss: {policy_loss}, value loss: {value_loss}, regularization loss: {reg_loss}".format(
                            policy_loss=policy_loss, value_loss=value_loss, reg_loss=reg_loss
                        )
                    )

                    if total_updates % self._process_config.target_network_update_frequency == 0:
                        self._copy_weights_from_train_to_decision()

                if i_step >= self._process_config.max_ep_length or game_over:
                    break

            # print("finished episode")
            reward_sums.append(self._game_world.player.reward_sum)

            if i_episode % 10 == 0 and i_episode != 0:
                tf.logging.info(
                    "Total steps: {total_steps}, Mean reward sum: {mean_reward_sum}, Temperature: {temp}".format(
                        total_steps=total_steps,
                        mean_reward_sum=np.mean(reward_sums[-10:]),
                        temp=self._current_exploration_temperature,
                    )
                )
                reward_sums.clear()

            if save_path is not None and i_episode % 100 == 0:
                self._saver.save(self._session, "{save_path}/model-{i_episode}.ckpt".format(save_path=save_path,
                                                                                            i_episode=i_episode))
                tf.logging.info("Saved model at episode {i_episode}.".format(i_episode=i_episode))

        self._saver.save(self._session, "{save_path}/model-final.ckpt".format(save_path=save_path))

    def get_world(self) -> World:
        return self._game_world


Exp = namedtuple("Exp", "chosen_action value reward inputs")


class DeepQLearnerWithExperienceReplay:
    _game_world = ...  # type: World
    _output_angles = ...  # type: np.ndarray
    _session = ...  # type: tf.Session
    _time_between_actions_s = ...  # type: float
    _process_config = ...  # type: LearningProcessConfig
    _n_steps_back = ...  # type: int

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

    def _apply_action_get_reward(self, action_index: int) -> float:
        angle, movement = self._get_angle_movement(action_index)
        return self._game_world.update_world_and_player_and_get_reward(self._time_between_actions_s, angle, movement)

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

        hidden_dense0 = Dense(units=self._n_sensor_inputs , activation="relu", name="hidden_dense1")
        x = hidden_dense0(concatenated_input)

        hidden_dense1 = Dense(units=self._n_sensor_inputs, activation="relu", name="hidden_dense1")
        x = hidden_dense1(x)

        hidden_dense2 = Dense(units=self._n_sensor_inputs // 2, activation="relu", name="hidden_dense2")
        x = hidden_dense2(x)

        hidden_dense3 = Dense(units=self._n_sensor_inputs // 2, activation="relu", name="hidden_dense3")
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

    def apply_action_from_input(self,
                                prev_sensor_states,
                                prev_heat_states) -> Tuple[np.ndarray, np.ndarray]:

        assert len(prev_sensor_states) == len(prev_heat_states) == self._n_steps_back - 1

        current_sensors = self.get_sensor_input()
        current_heat = self.get_heat_input()

        input_sensors = np.hstack((*prev_sensor_states, current_sensors)).reshape((1, *self._sensor_state_shape))
        input_heat = np.hstack((*prev_heat_states, current_heat)).reshape((1, *self._heat_state_shape))

        qualities = self._session.run(self._actions_qualities_tensor, feed_dict={
            self._sensor_input_tensor: input_sensors,
            self._heat_input_tensor: input_heat,
        })[0]
        action_index = np.argmax(qualities[:-1])  # type: int

        # action_index = self._session.run(self._action_index_tensor, feed_dict={
        #     self._sensor_input_tensor: input_sensors,
        #     self._heat_input_tensor: input_heat,
        # })[0]

        self._apply_action_get_reward(action_index)

        return current_sensors, current_heat

    def load_model(self, path):
        tf.logging.info("Loading model...")
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
        ep_buf = EpisodeBuffer(self._state_shapes, self._process_config.buffer_size, np.int32)
        # ep_buf_simple = EpisodeBufferSimple(self._process_config.buffer_size)
        reward_sums = []

        for i_episode in range(self._process_config.n_training_episodes):
            self._game_world.reset()
            current_sensor_state = np.tile(self.get_sensor_input(), self._n_steps_back)
            current_heat_state = np.tile(self.get_heat_input(), self._n_steps_back)
            # current_position_state = np.reshape(self._get_position_input(), (1, *self._position_state_shape))

            for i_step in range(self._process_config.max_ep_length):

                total_steps += 1
                if not pre_trained and total_steps > self._process_config.pre_train_steps:
                    pre_trained = True
                    tf.logging.info("Pre-training ended.")

                if total_steps < self._process_config.pre_train_steps or np.random.rand(1) < random_action_prob:
                    action_index = np.array([np.random.randint(0, self._n_actions)])
                else:
                    action_index = self._session.run(self._action_index_tensor, feed_dict={
                        self._sensor_input_tensor: current_sensor_state.reshape(1, *self._sensor_state_shape),
                        self._heat_input_tensor: current_heat_state.reshape(1, *self._heat_state_shape),
                        # self._position_input_tensor: current_position_state,
                    })

                skipped_frames_reward_sum = 0
                for _ in range(self._process_config.n_skipped_frames):
                    skipped_frames_reward_sum += self._apply_action_get_reward(action_index[0])
                game_over = self._game_world.game_over

                step_state = self.get_sensor_input()
                next_sensor_state = np.hstack((
                    current_sensor_state[self._n_sensors * self._n_sensor_types:],
                    step_state
                ))
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
                    np.array([skipped_frames_reward_sum], dtype=np.float32),
                    np.array([game_over], dtype=np.bool),
                ))
                # ep_buf_simple.add_experience(Experience(
                #     [np.array(current_sensor_state), np.array(current_heat_state)],
                #     [np.array(next_sensor_state), np.array(next_heat_state)],
                #     action_index[0],
                #     reward,
                #     game_over,
                # ))
                # print(
                #     ep_buf._reward_buffer[max(0, ep_buf._current_ptr - ep_buf._buffer_size): ep_buf._current_ptr]
                #     .reshape(
                #         (min(ep_buf._current_ptr, ep_buf._buffer_size), self._n_steps_back)
                #     )
                # )
                # [buffer_entry.state for buffer_entry in ep_buf_simple._buffer]
                # print(np.array([buffer_entry.reward for buffer_entry in ep_buf_simple._buffer]))
                # print(ep_buf_simple._buffer[0:len(ep_buf_simple._buffer)].state[0].reshape(
                #     (self._n_steps_back, self._n_sensors, self._n_sensor_types)
                # ))

                current_sensor_state[:] = next_sensor_state
                current_heat_state[:] = next_heat_state

                if total_steps > self._process_config.pre_train_steps:
                    if random_action_prob > self._process_config.end_random_action_prob:
                        random_action_prob -= step_drop

                    if total_steps % self._process_config.update_frequency == 0:
                        train_experiences = ep_buf.sample(self._process_config.replay_size)
                        # # train_experiences = ep_buf.sample(self._process_config.replay_size)
                        # indices = ep_buf.get_indices(self._process_config.replay_size)
                        # train_experiences = ep_buf.sample_indices(indices)
                        # experiences_list = ep_buf_simple.sample_indices(
                        #     indices - (ep_buf._current_ptr - ep_buf._buffer_size)
                        # )
                        # other_experiences = experience_list_to_experiences(self._state_shapes, experiences_list)
                        # assert_experiences_equals(train_experiences, other_experiences)

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
                        action_indices = train_experiences.actions
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

            reward_sums.append(self._game_world.player.reward_sum)

            if i_episode % 10 == 0 and i_episode != 0:
                tf.logging.info(
                    "Total steps: {total_steps}, Mean reward sum: {mean_reward_sum}, e = {random_action_prob}".format(
                        total_steps=total_steps,
                        mean_reward_sum=np.mean(reward_sums[-10:]),
                        random_action_prob=random_action_prob
                    )
                )
                reward_sums.clear()

            if save_path is not None and i_episode % 100 == 0 and i_episode != 0:
                self._saver.save(self._session, "{save_path}/model-{i_episode}.ckpt".format(save_path=save_path,
                                                                                            i_episode=i_episode))
                tf.logging.info("Saved model at episode {i_episode}.".format(i_episode=i_episode))

        self._saver.save(self._session, "{save_path}/model-final.ckpt".format(save_path=save_path))
