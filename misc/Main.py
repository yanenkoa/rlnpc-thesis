from math import pi

import numpy as np
import tensorflow as tf

from misc.Controllers import KeyboardController
from misc.InputDevice import InputDevice
from misc.Rendering import RenderWorld
from trainer.Configs import config_one
from trainer.GameObjects import World
from trainer.RL import DeepQLearnerWithExperienceReplay, LearningProcessConfig, ActorCriticRecurrentLearner


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

    # cloud_ml_training(config_one())

    # config = LearningProcessConfig(
    #     replay_size=4 * 64,
    #     update_frequency=2 * 64,
    #     reward_discount_coef=0.9,
    #     start_random_action_prob=1.,
    #     end_random_action_prob=0.1,
    #     annealing_steps=int(1e6),
    #     n_training_episodes=10000,
    #     pre_train_steps=5000,
    #     max_ep_length=(2 * 60 * 30),
    #     buffer_size=2000,
    #     n_skipped_frames=4,
    # )
    # learner = ActorCriticRecurrentLearner(
    #     world,
    #     tf.Session(),
    #     1. / 60,
    #     5,
    #     config
    # )
    # learner.initialize()
    #
    # render = RenderWorld(world)
    # render.start_drawing()
    # learner.loop(render.update)


if __name__ == '__main__':
    main()
