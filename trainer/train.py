from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from trainer.Configs import config_one
from trainer.GameObjects import World
from trainer.RL import DeepQLearnerWithExperienceReplay, LearningProcessConfig, ActorCriticRecurrentLearner


def cloud_ml_training(world_config, path: str):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info("Writing to {path}".format(path=path))

    world = World(*world_config)
    config = LearningProcessConfig(
        replay_size=4 * 64,
        update_frequency=2 * 64,
        reward_discount_coef=0.9,
        start_random_action_prob=1.,
        end_random_action_prob=0.1,
        annealing_steps=int(1e7),
        n_training_episodes=10000,
        pre_train_steps=10000,
        max_ep_length=(2 * 60 * 30),
        buffer_size=10000,
    )
    session = tf.Session()
    output_angles = np.linspace(-np.pi / 2, np.pi / 2, 16, False)
    learner = DeepQLearnerWithExperienceReplay(world, output_angles, session, 1. / 15, 5, config)
    learner.initialize()

    tf.logging.info("Shall we?")
    learner.train(path)


def ac_training(world_config, path: str):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info("Writing to {path}".format(path=path))

    world = World(*world_config)

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
    learner.initialize()

    learner.train(path)


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    argparser = ArgumentParser()
    argparser.add_argument("--job-dir", default=None)

    args = argparser.parse_args()

    ac_training(config_one(), args.job_dir)


if __name__ == '__main__':
    main()
