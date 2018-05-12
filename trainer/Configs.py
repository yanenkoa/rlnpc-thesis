import numpy as np

from trainer.GameObjects import Player, RectangleWall, GoldChest, Portal, ProximitySensors, HeatSource
from trainer.Util import Vector2, RectangleAABB


def config_empty():
    width = 1000
    height = 1000
    player = Player(Vector2(width / 2, height / 2), np.pi / 2, 100)
    walls = [
        # RectangleWall(RectangleAABB(Vector2(0, 0), Vector2(400, 400))),
        # RectangleWall(RectangleAABB(Vector2(600, 0), Vector2(1000, 400))),
        # RectangleWall(RectangleAABB(Vector2(0, 600), Vector2(1000, 800))),
    ]
    left_wall = RectangleWall(RectangleAABB(Vector2(-100, 0), Vector2(0, height)))
    top_wall = RectangleWall(RectangleAABB(Vector2(0, height), Vector2(width, height + 100)))
    right_wall = RectangleWall(RectangleAABB(Vector2(width, 0), Vector2(width + 100, height)))
    bottom_wall = RectangleWall(RectangleAABB(Vector2(0, -100), Vector2(width, 0)))
    walls.extend([left_wall, top_wall, right_wall, bottom_wall])
    gold_chests = [
        GoldChest(500, Vector2(100, height / 2))
    ]
    heat_sources = []
    portal = Portal(Vector2(600, 500))
    proximity_sensors_np = ProximitySensors(
        player,
        np.linspace(-np.pi / 2, np.pi / 2, 256, False),
        600,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def config_one():
    width = 1000
    height = 1000
    player = Player(Vector2(50, 50), np.pi / 2, 300)
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
    left_wall = RectangleWall(RectangleAABB(Vector2(-100, 0), Vector2(0, height)))
    top_wall = RectangleWall(RectangleAABB(Vector2(0, height), Vector2(width, height + 100)))
    right_wall = RectangleWall(RectangleAABB(Vector2(width, 0), Vector2(width + 100, height)))
    bottom_wall = RectangleWall(RectangleAABB(Vector2(0, -100), Vector2(width, 0)))
    walls.extend([left_wall, top_wall, right_wall, bottom_wall])
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

    proximity_sensors_np = ProximitySensors(
        player,
        np.linspace(-np.pi, np.pi, 128, False),
        500,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def config_two():
    width = 1000
    height = 1000
    player = Player(Vector2(width / 2, 50), np.pi / 2, 300)
    walls = [
        RectangleWall(RectangleAABB(Vector2(0, 0), Vector2(400, 400))),
        RectangleWall(RectangleAABB(Vector2(600, 0), Vector2(1000, 400))),
        RectangleWall(RectangleAABB(Vector2(0, 600), Vector2(1000, 650))),
    ]
    left_wall = RectangleWall(RectangleAABB(Vector2(-100, 0), Vector2(0, height)))
    top_wall = RectangleWall(RectangleAABB(Vector2(0, height), Vector2(width, height + 100)))
    right_wall = RectangleWall(RectangleAABB(Vector2(width, 0), Vector2(width + 100, height)))
    bottom_wall = RectangleWall(RectangleAABB(Vector2(0, -100), Vector2(width, 0)))
    walls.extend([left_wall, top_wall, right_wall, bottom_wall])
    gold_chests = [
        GoldChest(500, Vector2(100, height / 2))
    ]
    heat_sources = []
    portal = Portal(Vector2(800, 500))
    proximity_sensors_np = ProximitySensors(
        player,
        np.linspace(-np.pi / 2, np.pi / 2, 64, False),
        600,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np
