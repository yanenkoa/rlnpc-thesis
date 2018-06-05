from math import pi

import numpy as np
import tensorflow as tf

from misc.Controllers import KeyboardController
from misc.InputDevice import InputDevice
from misc.Rendering import RenderWorld
from trainer.Configs import config_one, config_two, get_config
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
    learner._copy_weights_from_train_to_decision()

    render_world = RenderWorld(learner.get_world())
    render_world.start_drawing()
    learner.loop(temp, render_world.update)


def simulate_keyboard(world_config):
    world = World(*world_config)
    input_device = InputDevice()

    controller = KeyboardController(5, input_device, world)
    controller.loop()


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    lp_config, net_config = get_config("gs://eneka-storage/configs/rl_config_even_lower_lr_clipnorm.json")
    learner = initialize_ac_learner(config_one(), lp_config, net_config)
    load_path = "gs://eneka-storage/a2c_global_avg_std_3"
    n_iter = 400
    load_n_loop(learner, load_path, n_iter, 0.1)
    # simulate_keyboard(config_one())


if __name__ == '__main__':
    main()
