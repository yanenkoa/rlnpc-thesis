from time import time

from pynput.keyboard import Key

from GameObjects import World, Player, PlayerMovementDirection
from InputDevice import InputDevice
from Rendering import RenderWorld


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

    def _update_player(self, elapsed_time_s) -> None:

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

    def start_rendering(self) -> None:
        if self._input_device.is_key_down("v"):
            self._render_world.start_drawing()

    def reset(self) -> None:
        if self._input_device.is_key_down("r"):
            self._world.reset()

    def loop(self):
        while True:
            self.reset()
            self.start_rendering()
            elapsed_time_s = self._get_elapsed_time_s()
            self._update_player(elapsed_time_s)
            self._world.update_state(elapsed_time_s)
            self._render_world.update()
            if self._input_device.is_key_down(Key.esc):
                break
