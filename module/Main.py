from math import pi

import numpy as np
import tensorflow as tf

from module.Controllers import KeyboardController
from module.GameObjects import Player, RectangleWall, GoldChest, HeatSource, Portal, World, ProximitySensors
# from InputDevice import InputDevice
from module.InputDevice import InputDevice
from module.RL import DeepQLearnerWithExperienceReplay, LearningProcessConfig
# from Rendering import RenderWorld
from module.Util import Vector2, RectangleAABB


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
    player = Player(Vector2(50, 300), -pi / 2, 300)
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
        np.linspace(-np.pi / 2, np.pi / 2, 40, False),
        500,
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
        np.linspace(-np.pi / 2, np.pi / 2, 64, False),
        600,
        walls,
        gold_chests,
        portal
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def cloud_ml_training(world_config):
    world = World(*world_config)
    config = LearningProcessConfig(
        replay_size=4 * 64,
        update_frequency=2 * 64,
        reward_discount_coef=0.9,
        start_random_action_prob=1.,
        end_random_action_prob=0.1,
        annealing_steps=10000,
        n_training_episodes=10000,
        pre_train_steps=5000,
        max_ep_length=(2 * 60 * 30),
        buffer_size=2000,
    )
    session = tf.Session()
    output_angles = np.linspace(-np.pi / 2, np.pi / 2, 16, False)
    learner = DeepQLearnerWithExperienceReplay(world, output_angles, session, 1. / 15, 5, config)
    learner.initialize()

    learner.train("data")


def main():
    # world = World(*config_one())
    #
    # turn_rate_ps = pi / 0.8
    # input_device = InputDevice()
    # controller = KeyboardController(turn_rate_ps, input_device, world)
    #
    # controller.loop()

    # config = LearningProcessConfig(
    #     replay_size=4 * 64,
    #     update_frequency=2 * 64,
    #     reward_discount_coef=0.9,
    #     start_random_action_prob=1.,
    #     end_random_action_prob=0.1,
    #     annealing_steps=10000,
    #     n_training_episodes=10000,
    #     pre_train_steps=5000,
    #     max_ep_length=(2 * 60 * 30),
    #     buffer_size=2000,
    # )
    # session = tf.Session()
    # output_angles = np.linspace(-np.pi / 2, np.pi / 2, 16, False)
    # learner = DeepQLearnerWithExperienceReplay(world, output_angles, session, 1. / 15, 5, config)
    # learner.initialize()
    # # render_world.start_drawing()
    # # input_device = InputDevice()
    #
    # # def render_stuff():
    # #     nonlocal render_world, input_device
    # #     if input_device.is_key_down("s"):
    # #         render_world.start_drawing()
    # #     if input_device.is_key_down("d"):
    # #         render_world.stop_drawing()
    # #     render_world.update()
    #
    # render_world: RenderWorld = RenderWorld(world)
    # learner.load_model("/media/d/learning_data")
    #
    # init_sensor = learner.get_sensor_input()
    # init_heat = learner.get_heat_input()
    #
    # prev_sensor_states = deque(
    #     [np.array(init_sensor), np.array(init_sensor), np.array(init_sensor), np.array(init_sensor)]
    # )
    # prev_heat_states = deque(
    #     [np.array(init_heat), np.array(init_heat), np.array(init_heat), np.array(init_heat)]
    # )
    #
    # render_world.start_drawing()
    # while True:
    #     learner.apply_action_from_input(prev_sensor_states, prev_heat_states)
    #     cur_sensor = learner.get_sensor_input()
    #     cur_heat = learner.get_heat_input()
    #     prev_sensor_states.popleft()
    #     prev_sensor_states.append(cur_sensor)
    #     prev_heat_states.popleft()
    #     prev_heat_states.append(cur_heat)
    #     world.player.reset_reward_after_step()
    #     render_world.update()
    #
    # # while True:
    # #     learner.apply_action_from_network()
    # #     render_world.update()
    #
    # # controller.loop()
    # # learner.train("/media/d/learning_data", render_world.update)

    # with tf.device("/cpu:0"):
    cloud_ml_training(config_one())


if __name__ == '__main__':
    main()
