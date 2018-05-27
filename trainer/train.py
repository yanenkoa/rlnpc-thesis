from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from trainer.Configs import config_one, get_config
from trainer.GameObjects import World
from trainer.RL import DeepQLearnerWithExperienceReplay, LearningProcessConfig, ActorCriticRecurrentLearner, \
    NetworkConfig


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


def initialize_ac_learner(world_config,
                          process_config: LearningProcessConfig,
                          network_config: NetworkConfig) -> ActorCriticRecurrentLearner:
    world = World(*world_config)
    learner = ActorCriticRecurrentLearner(
        world,
        tf.Session(),
        network_config,
        process_config
    )

    learner.initialize_a2c()
    return learner


def ac_training(world_config, dump_path: str, process_config: LearningProcessConfig, network_config: NetworkConfig):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info("Writing to {path}".format(path=dump_path))
    learner = initialize_ac_learner(world_config, process_config, network_config)
    learner.train(dump_path)


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    argparser = ArgumentParser()
    argparser.add_argument("--job-dir", default=None)
    argparser.add_argument("--config-dir", default=None)

    args = argparser.parse_args()

    process_config, network_config = get_config(args.config_dir)
    ac_training(config_one(), args.job_dir, process_config, network_config)


if __name__ == '__main__':
    main()
