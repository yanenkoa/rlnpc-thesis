from math import pi

import numpy as np

from Controllers import KeyboardController
from GameObjects import Player, RectangleWall, GoldChest, HeatSource, Portal, ProximitySensor, World
from InputDevice import InputDevice
from Util import Vector2, RectangleAABB


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
                         for angle in np.linspace(0, 2 * np.pi, 40, False)]
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors


def main():
    world = World(*config_two())

    turn_rate_ps = pi / 0.8
    input_device = InputDevice()
    controller = KeyboardController(turn_rate_ps, input_device, world)
    controller.loop()


if __name__ == '__main__':
    main()
