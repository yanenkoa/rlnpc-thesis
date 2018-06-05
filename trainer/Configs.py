import json
from pprint import pprint
from typing import Tuple
from tensorflow.python.lib.io import file_io

import numpy as np

from trainer.GameObjects import Player, RectangleWall, GoldChest, Portal, ProximitySensors, HeatSource
from trainer.RL import NetworkConfig, ConvConfig, LSTMConfig, DenseConfig, LearningProcessConfig, \
    network_config_to_dict, dict_to_network_config
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
    player = Player(Vector2(950, 500), np.pi / 2, 300)
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
         GoldChest(35, Vector2(950, 50)),
         GoldChest(25, Vector2(50, 500)),
         GoldChest(30, Vector2(500, 950)),
         GoldChest(50, Vector2(50, 50)),
         GoldChest(30, Vector2(250, 50)),
         GoldChest(10, Vector2(750, 50)),
         GoldChest(5, Vector2(350, 450)),
         GoldChest(20, Vector2(50, 850)),
    ]
    heat_sources = [
        HeatSource(10, Vector2(450, 450), walls),
    ]
    portal = Portal(Vector2(800, 750))

    proximity_sensors_np = ProximitySensors(
        player,
        np.linspace(-np.pi, np.pi, 60, False),
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
        RectangleWall(RectangleAABB(Vector2(0, 600), Vector2(800, 650))),
        RectangleWall(RectangleAABB(Vector2(750, 650), Vector2(800, 1000))),
    ]
    left_wall = RectangleWall(RectangleAABB(Vector2(-100, 0), Vector2(0, height)))
    top_wall = RectangleWall(RectangleAABB(Vector2(0, height), Vector2(width, height + 100)))
    right_wall = RectangleWall(RectangleAABB(Vector2(width, 0), Vector2(width + 100, height)))
    bottom_wall = RectangleWall(RectangleAABB(Vector2(0, -100), Vector2(width, 0)))
    walls.extend([left_wall, top_wall, right_wall, bottom_wall])
    gold_chests = [
        GoldChest(250, Vector2(100, height / 2)),
    ]
    heat_sources = []
    portal = Portal(Vector2(900, 900))
    proximity_sensors_np = ProximitySensors(
        player,
        np.linspace(-np.pi, np.pi, 60, False),
        700,
        walls,
        gold_chests,
        portal,
    )
    return width, height, player, walls, gold_chests, heat_sources, portal, proximity_sensors_np


def rl_config() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=1000,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.5,
        framerate=framerate,
        regularization_loss_coef=1e-3,
        learning_rate=0.0005,
        clip_norm=1e8,
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=437,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_2",
            ),
        ],
    )

    return lp_config, net_conf


def rl_config_two() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=20,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.5,
        framerate=framerate,
        regularization_loss_coef=1,
        learning_rate=0.0001,
        clip_norm=1e8,
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=437,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_2",
            ),
        ],
    )

    return lp_config, net_conf


def rl_config_larger_lr_lower_clipnorm() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=20,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.5,
        framerate=framerate,
        regularization_loss_coef=1,
        learning_rate=0.001,
        clip_norm=40
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=437,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_2",
            ),
        ],
    )

    return lp_config, net_conf


def rl_config_low_lr_low_clipnorm() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=20,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.1,
        framerate=framerate,
        regularization_loss_coef=1,
        learning_rate=0.000001,
        clip_norm=5
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=437,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_2",
            ),
        ],
    )

    return lp_config, net_conf


def rl_config_even_lower_lr_clipnorm() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=20,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.1,
        framerate=framerate,
        regularization_loss_coef=1,
        learning_rate=0.00000001,
        clip_norm=1
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=437,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_2",
            ),
        ],
    )

    return lp_config, net_conf


def rl_config_alt_3() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 10
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length * 2
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=20,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.1,
        framerate=framerate,
        regularization_loss_coef=1e-1,
        learning_rate=0.0001,
        clip_norm=200
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv2",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=218,
                name="lstm_layer_1",
            ),
            LSTMConfig(
                units=218,
                name="lstm_layer_2",
            ),
            LSTMConfig(
                units=218,
                name="lstm_layer_2",
            ),
        ],
        dense_configs=[],
    )

    return lp_config, net_conf


def rl_config_shallower() -> Tuple[LearningProcessConfig, NetworkConfig]:
    frames_in_second = 60
    n_skipped_frames = 15
    max_minutes = 5
    framerate = 1. / frames_in_second
    max_ep_length = max_minutes * 60 * frames_in_second // n_skipped_frames
    update_frequency = max_ep_length // 5
    lp_config = LearningProcessConfig(
        replay_size=None,
        update_frequency=update_frequency,
        reward_discount_coef=0.9,
        start_random_action_prob=None,
        end_random_action_prob=None,
        annealing_steps=None,
        n_training_episodes=5000,
        pre_train_steps=None,
        max_ep_length=max_ep_length,
        buffer_size=None,
        n_skipped_frames=n_skipped_frames,
        target_network_update_frequency=1000,
        initial_temperature=10,
        temp_coef=0.00001,
        min_temperature=0.5,
        framerate=framerate,
        regularization_loss_coef=1e-1,
        learning_rate=0.001,
        clip_norm=1e8,
    )
    net_conf = NetworkConfig(
        window_size=7,
        n_output_angles=8,
        conv_configs=[
            ConvConfig(
                filters=16,
                activation="prelu",
                name="conv1",
            ),
        ],
        lstm_configs=[
            LSTMConfig(
                units=437,
                name="lstm_layer_1",
            ),
        ],
        dense_configs=[
            DenseConfig(
                units=218,
                activation="prelu",
                name="dense_layer_1",
            ),
        ],
    )

    return lp_config, net_conf


def get_config(path: str) -> Tuple[LearningProcessConfig, NetworkConfig]:
    with file_io.FileIO(path, mode="r") as f:
        json_config = json.load(f)

    pprint(json_config)

    learning_process_config = LearningProcessConfig(**json_config["learning_process_config"])
    network_config = dict_to_network_config(json_config["network_config"])

    return learning_process_config, network_config


def dump_config(lpc: LearningProcessConfig, nc: NetworkConfig, path: str) -> None:
    json_lpc = dict(lpc._asdict())
    json_nc = network_config_to_dict(nc)
    json_result = {
        "learning_process_config": json_lpc,
        "network_config": json_nc,
    }

    import pathlib
    pathlib.Path(path.rpartition("/")[0]).mkdir(parents=True, exist_ok=True)

    with file_io.FileIO(path, mode="w+") as f:
        json.dump(json_result, f, indent=2, separators=(',', ': '))


def main():
    config_funcs = [rl_config, rl_config_shallower, rl_config_two, rl_config_larger_lr_lower_clipnorm,
                    rl_config_low_lr_low_clipnorm, rl_config_even_lower_lr_clipnorm, rl_config_alt_3]
    print("Choose a config to dump:")
    for i, config_func in enumerate(config_funcs):
        print("{}: {}".format(i, config_func.__name__))

    i_config = int(input())
    lpc, network_config = config_funcs[i_config]()

    path = "gs://eneka-storage/configs/{}.json".format(config_funcs[i_config].__name__)
    print("Writing to {}".format(path))

    dump_config(lpc, network_config, path)


if __name__ == '__main__':
    main()
