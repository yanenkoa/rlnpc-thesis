from time import time, sleep
from typing import Tuple

from pynput.keyboard import Key

from trainer.GameObjects import World, Player, PlayerMovementDirection
from misc.InputDevice import InputDevice
from misc.Rendering import RenderWorld


class KeyboardController:
    _turn_rate_ps = ...  # type: float
    _input_device = ...  # type: InputDevice
    _world = ...  # type: World

    _player = ...  # type: Player
    _render_world = ...  # type: RenderWorld
    _last_update_s = ...  # type: float

    def __init__(self, turn_rate_ps: float, input_device: InputDevice, world: World):
        self._turn_rate_ps = turn_rate_ps
        self._input_device = input_device
        self._world = world
        self._player = self._world.player
        self._render_world = RenderWorld(self._world)
        self._last_update_s = time()

    def _get_angle_increment(self, elapsed_time_s: float, clockwise: bool) -> float:
        return self._turn_rate_ps * elapsed_time_s * (-1 if clockwise else 1)

    def _get_elapsed_time_s(self) -> float:
        cur_time_s = time()
        elapsed_time_s = cur_time_s - self._last_update_s
        self._last_update_s = cur_time_s
        return elapsed_time_s

    def _get_player_update(self, elapsed_time_s) -> Tuple[float, PlayerMovementDirection]:
        if self._input_device.is_key_down("d"):
            angle_increment = self._get_angle_increment(elapsed_time_s, True)
        elif self._input_device.is_key_down("a"):
            angle_increment = self._get_angle_increment(elapsed_time_s, False)
        else:
            angle_increment = 0.0
        new_angle = self._player.angle + angle_increment

        if self._input_device.is_key_down("w"):
            movement_direction = PlayerMovementDirection.FORWARD
        elif self._input_device.is_key_down("s"):
            movement_direction = PlayerMovementDirection.BACKWARD
        else:
            movement_direction = PlayerMovementDirection.NONE

        return new_angle, movement_direction

    def _start_rendering(self) -> None:
        if self._input_device.is_key_down("v"):
            self._render_world.start_drawing()

    def _reset(self) -> None:
        if self._input_device.is_key_down("r"):
            self._world.reset()

    def loop(self):
        while True:
            sleep(1. / 60)

            self._reset()
            self._render_world.start_drawing()

            elapsed_time_s = self._get_elapsed_time_s()
            new_angle, movement_direction = self._get_player_update(elapsed_time_s)
            self._world.update_world_and_player_and_get_reward(elapsed_time_s, new_angle, movement_direction)

            self._render_world.update()
            if self._input_device.is_key_down(Key.esc):
                break


# class RLController:
#     _world = ...  # type: World
#     _fps = ...  # type: int
#
#     _time_between_frames = ...  # type: float
#     _window_size = ...  # type: int
#     _learner = ...  # type: DeepQLearnerWithExperienceReplay
#     _render_world = ...  # type: RenderWorld
#     _last_update_s = ...  # type: float
#
#     def __init__(self, world: World, fps: int = 30):
#         self._world = world
#         self._fps = fps
#
#         self._time_between_frames = 1 / self._fps
#         self._window_size = 7
#         fake_process_config = namedtuple("SomeTuple", ["reward_discount_coef"])(0.9)
#         self._learner = DeepQLearnerWithExperienceReplay(world, 128, tf.Session(), self._window_size, 1. / 30,
#                                                          fake_process_config)
#         self._render_world = RenderWorld(self._world)
#         self._last_update_s = time()
#
#     def initialize(self) -> None:
#         self._learner.initialize()
#         self._learner.get_next_action()
#
#     def _get_elapsed_time_s(self) -> float:
#         cur_time_s = time()
#         elapsed_time_s = cur_time_s - self._last_update_s
#         self._last_update_s = cur_time_s
#         return elapsed_time_s
#
#     def loop(self) -> None:
#         i = 0
#         while True:
#             i += 1
#             print(i)
#             self._render_world.start_drawing()
#             elapsed_time_s = self._get_elapsed_time_s()
#             self._learner.apply_action(elapsed_time_s)
#             if elapsed_time_s < self._time_between_frames:
#                 sleep(self._time_between_frames - elapsed_time_s)
#             self._render_world.update()
