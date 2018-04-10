from math import pi

import numpy as np
import tensorflow as tf

from GameObjects import Player, RectangleWall, GoldChest, HeatSource, Portal, World, ProximitySensors
from InputDevice import InputDevice
from RL import Learner, LearningProcessConfig
from Rendering import RenderWorld
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
        np.linspace(0, 2 * np.pi, 512, False),
        600,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def config_two():
    width = 1000
    height = 1000
    player = Player(Vector2(width / 2, 50), pi / 2, 300)
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
        np.linspace(0, 2 * np.pi, 256, False),
        500,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def config_empty():
    width = 1000
    height = 1000
    player = Player(Vector2(width / 2, height / 2), pi / 2, 100)
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
        np.linspace(0, 2 * np.pi, 256, False),
        600,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def main():
    world = World(*config_two())

    # turn_rate_ps = pi / 0.8
    # input_device = InputDevice()
    # controller = KeyboardController(turn_rate_ps, input_device, world)
    # controller.loop()

    config = LearningProcessConfig(
        replay_size=128,
        update_frequency=16,
        reward_discount_coef=0.9,
        start_random_action_prob=1.,
        end_random_action_prob=0.1,
        annealing_steps=10000,
        n_training_episodes=10000,
        pre_train_steps=10000,
        max_ep_length=(5 * 60 * 30),
    )
    session = tf.Session()
    learner = Learner(world, 128, session, 7, 1. / 30, config)
    learner.initialize()
    # render_world = RenderWorld(world)
    # render_world.start_drawing()
    render_world: RenderWorld = RenderWorld(world)
    input_device = InputDevice()

    def render_stuff():
        nonlocal render_world, input_device
        if input_device.is_key_down("s"):
            render_world.start_drawing()
        elif input_device.is_key_down("d"):
            render_world.stop_drawing()
        render_world.update()

    learner.train("data", render_stuff)


if __name__ == '__main__':
    main()
