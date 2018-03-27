from math import pi
from time import time

import numpy as np
from pynput.keyboard import Key

from GameObjects import World, Player, RectangleWall, GoldChest, HeatSource, Portal, PlayerMovementDirection, \
    ProximitySensor
from InputDevice import InputDevice
from Rendering import RenderWorld
from Util import Vector2, RectangleAABB


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

    def loop(self):
        while True:
            elapsed_time_s = self._get_elapsed_time_s()
            self._update_player(elapsed_time_s)
            self._world.update_state(elapsed_time_s)
            self._render_world.update()
            if self._input_device.is_key_down(Key.esc):
                break


def config_one():
    width = 1000
    height = 1000
    player = Player(Vector2(width - 50, 50), pi / 2, 300)
    walls = [
        RectangleWall(RectangleAABB(Vector2(800, 0.0), Vector2(900, 400))),
        RectangleWall(RectangleAABB(Vector2(500, 400), Vector2(900, 500))),
        RectangleWall(RectangleAABB(Vector2(500, 0.0), Vector2(600, 300))),
        RectangleWall(RectangleAABB(Vector2(100, 0.0), Vector2(200, 300))),
        RectangleWall(RectangleAABB(Vector2(200, 100), Vector2(400, 200))),
        RectangleWall(RectangleAABB(Vector2(100, 300), Vector2(300, 400))),
        RectangleWall(RectangleAABB(Vector2(100, 300), Vector2(300, 400))),
        RectangleWall(RectangleAABB(Vector2(600, 600), Vector2(950, 700))),
        RectangleWall(RectangleAABB(Vector2(850, 700), Vector2(950, 950))),
        RectangleWall(RectangleAABB(Vector2(600, 700), Vector2(700, 850))),
        RectangleWall(RectangleAABB(Vector2(0.0, 700), Vector2(450, 800))),
    ]
    gold_chests = [
        GoldChest(500, Vector2(50, 50)),
        GoldChest(300, Vector2(250, 50)),
        GoldChest(100, Vector2(750, 50)),
        GoldChest(50, Vector2(350, 450)),
        GoldChest(200, Vector2(50, 850)),
    ]
    heat_sources = [
        HeatSource(1000, Vector2(450, 450), walls),
    ]
    portal = Portal(Vector2(800, 750))
    proximity_sensors = [ProximitySensor(player, angle, 200, walls, gold_chests, portal)
                         for angle in np.linspace(0, 2 * np.pi, 64, False)]
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors


def config_two():
    width = 1000
    height = 1000
    player = Player(Vector2(width / 2, 100), pi / 2, 300)
    walls = [
        RectangleWall(RectangleAABB(Vector2(0, 0), Vector2(400, 400))),
        RectangleWall(RectangleAABB(Vector2(600, 0), Vector2(1000, 400))),
        RectangleWall(RectangleAABB(Vector2(0, 600), Vector2(1000, 800))),
    ]
    gold_chests = [
        GoldChest(500, Vector2(100, height / 2))
    ]
    heat_sources = []
    portal = Portal(Vector2(800, 500))
    proximity_sensors = [ProximitySensor(player, angle, 200, walls, gold_chests, portal)
                         for angle in np.linspace(0, 2 * np.pi, 64, False)]
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors


def main():
    world = World(*config_two())

    turn_rate_ps = pi / 0.8
    input_device = InputDevice()
    controller = KeyboardController(turn_rate_ps, input_device, world)
    controller.loop()


if __name__ == '__main__':
    main()
