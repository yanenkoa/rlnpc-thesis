from math import pi

import numpy as np
import tensorflow as tf

from misc.Controllers import KeyboardController
from misc.InputDevice import InputDevice
from trainer.Configs import config_one
from trainer.GameObjects import World
from trainer.RL import DeepQLearnerWithExperienceReplay, LearningProcessConfig


def cloud_ml_training(world_config):
    world = World(*world_config)
    config = LearningProcessConfig(
        replay_size=4 * 64,
        update_frequency=2 * 64,
        reward_discount_coef=0.9,
        start_random_action_prob=1.,
        end_random_action_prob=0.1,
        annealing_steps=int(1e6),
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
    world = World(*config_one())

    turn_rate_ps = pi / 0.8
    input_device = InputDevice()
    controller = KeyboardController(turn_rate_ps, input_device, world)

    controller.loop()

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
