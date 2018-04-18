from collections import namedtuple
from time import time, sleep

import tensorflow as tf
from pynput.keyboard import Key

from module.GameObjects import World, Player, PlayerMovementDirection
from module.InputDevice import InputDevice
from module.RL import DeepQLearnerWithExperienceReplay
from module.Rendering import RenderWorld


class KeyboardController:
    _turn_rate_ps: float
    _input_device: InputDevice
    _world: World

    _player: Player
    _render_world: RenderWorld
    _last_update_s: float

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

    def _update_player(self, elapsed_time_s: float) -> None:

        if self._input_device.is_key_down("w"):
            self._world.move_player(elapsed_time_s, PlayerMovementDirection.FORWARD)
        elif self._input_device.is_key_down("s"):
            self._world.move_player(elapsed_time_s, PlayerMovementDirection.BACKWARD)
        else:
            self._world.move_player(elapsed_time_s, PlayerMovementDirection.NONE)

        if self._input_device.is_key_down("d"):
            angle_increment = self._get_angle_increment(elapsed_time_s, True)
        elif self._input_device.is_key_down("a"):
            angle_increment = self._get_angle_increment(elapsed_time_s, False)
        else:
            angle_increment = 0.0

        new_angle = self._player.angle + angle_increment
        self._world.update_player_angle(new_angle)

    def _start_rendering(self) -> None:
        if self._input_device.is_key_down("v"):
            self._render_world.start_drawing()

    def _reset(self) -> None:
        if self._input_device.is_key_down("r"):
            self._world.reset()

    def loop(self):
        while True:
            self._reset()
            # self._start_rendering()
            self._render_world.start_drawing()
            elapsed_time_s = self._get_elapsed_time_s()
            self._update_player(elapsed_time_s)
            self._world.update_state(elapsed_time_s)
            self._world.player.reset_reward_after_step()
            self._render_world.update()
            if self._input_device.is_key_down(Key.esc):
                break


class RLController:
    _world: World
    _fps: int

    _time_between_frames: float
    _window_size: int
    _learner: DeepQLearnerWithExperienceReplay
    _render_world: RenderWorld
    _last_update_s: float

    def __init__(self, world: World, fps: int = 30):
        self._world = world
        self._fps = fps

        self._time_between_frames = 1 / self._fps
        self._window_size = 7
        fake_process_config = namedtuple("SomeTuple", ["reward_discount_coef"])(0.9)
        self._learner = DeepQLearnerWithExperienceReplay(world, 128, tf.Session(), self._window_size, 1. / 30,
                                                         fake_process_config)
        self._render_world = RenderWorld(self._world)
        self._last_update_s = time()

    def initialize(self) -> None:
        self._learner.initialize()
        self._learner.get_next_action()

    def _get_elapsed_time_s(self) -> float:
        cur_time_s = time()
        elapsed_time_s = cur_time_s - self._last_update_s
        self._last_update_s = cur_time_s
        return elapsed_time_s

    def loop(self) -> None:
        i = 0
        while True:
            i += 1
            print(i)
            self._render_world.start_drawing()
            elapsed_time_s = self._get_elapsed_time_s()
            self._learner.apply_action(elapsed_time_s)
            if elapsed_time_s < self._time_between_frames:
                sleep(self._time_between_frames - elapsed_time_s)
            self._render_world.update()
