import argparse as ap
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.config import *
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_ocean.calculator_1d import Calculator1d
from alphagen_ocean.callbacks import CustomCallback
from alphagen_ocean.stock_data import ArgData

args = ap.ArgumentParser()
args.add_argument("--gpu", "-g", type=int, default=1)
args.add_argument("--name", "-n", type=str, default="ret1d")
args.add_argument("--kwargs", "-k", type=str, default="None")
args = args.parse_args()


DEVICE_MODEL = torch.device(f"cuda:{args.gpu}")
DEVICE_DATA = torch.device("cpu")
DEVICE_CALC = torch.device("cpu")


def main(
    seed: int = 0,
    instruments: str = "all",
    pool_capacity: int = 10,
    steps: int = 200_000,
):
    reseed_everything(seed)

    data_train = ArgData(
        start_time=20190103,
        end_time=20201231,
        device=DEVICE_DATA,
        max_backtrack_days=HORIZON_BACK,
    )
    data_valid = ArgData(
        start_time=20200101,
        end_time=20201231,
        device=DEVICE_DATA,
        max_backtrack_days=HORIZON_BACK,
    )
    data_test = ArgData(
        start_time=20210101,
        end_time=20211231,
        device=DEVICE_DATA,
        max_backtrack_days=HORIZON_BACK,
    )
    print("train days:", data_train.n_days)
    print("  val days:", data_valid.n_days)
    print(" test days:", data_test.n_days)

    calculator_train = Calculator1d(data_train, device=DEVICE_CALC)
    calculator_valid = Calculator1d(data_valid, device=DEVICE_CALC)
    calculator_test = Calculator1d(data_test, device=DEVICE_CALC)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=DEVICE_MODEL,
    )
    env = AlphaEnv(pool=pool, device=DEVICE_MODEL, print_expr=True)

    name_prefix = f"{instruments}_{pool_capacity}_{str(seed).zfill(4)}"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path="./checkpoints",
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=256,  # init 128
                dropout=0.1,
                device=DEVICE_MODEL,
            ),
        ),
        gamma=1.0,
        ent_coef=0.1,  # NOTE 1e-2
        batch_size=1024,  # 512
        tensorboard_log="./log",
        device=DEVICE_MODEL,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f"{timestamp}_{name_prefix}",
    )


if __name__ == "__main__":
    main(
        seed=random.randint(0, 9999),
        instruments=f"{args.name}_lexpr{str(MAX_EXPR_LENGTH).zfill(2)}_lopt{len(OPERATORS)}",
        pool_capacity=10,
        # steps=80_000,
        steps=250_000,
    )
