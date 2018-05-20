from math import pi

import numpy as np
import tensorflow as tf

from misc.Controllers import KeyboardController
from misc.InputDevice import InputDevice
from misc.Rendering import RenderWorld
from trainer.Configs import config_one, config_two
from trainer.GameObjects import World
from trainer.train import initialize_ac_learner
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


def load_n_loop(learner: ActorCriticRecurrentLearner, load_path: str, n_iter: int, temp: float) -> None:
    learner.load_model(load_path, n_iter)

    render_world = RenderWorld(learner.get_world())
    render_world.start_drawing()
    learner.loop(temp, render_world.update)


def main():
    learner = initialize_ac_learner(config_two())
    load_path = "gs://eneka-models/a2c_deeper_temp"
    n_iter = 200
    load_n_loop(learner, load_path, n_iter, 0.05)


if __name__ == '__main__':
    main()
