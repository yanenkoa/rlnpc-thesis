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

    # turn_rate_ps = pi / 0.8
    # input_device = InputDevice()
    # controller = KeyboardController(turn_rate_ps, input_device, world)
    # controller.loop()

    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 2
    framerate = 1. / frames_in_second

    config = LearningProcessConfig(
        replay_size=None,
        update_frequency=16,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_minutes * 60 * frames_in_second // n_skipped_frames,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
    )
    learner = ActorCriticRecurrentLearner(
        world,
        tf.Session(),
        32,
        framerate,
        7,
        config
    )
    learner.initialize_a2c()

    learner.load_model("gs://eneka-models/a2c_norm_rewards", 300)

    # learner.print_weights()

    render_world = RenderWorld(world)
    render_world.start_drawing()
    learner.loop(render_world.update)
    # learner.train(None, render_world.update)


if __name__ == '__main__':
    main()
