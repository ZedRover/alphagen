import argparse as ap
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from alphagen.config import *
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_ocean.calculator_t0 import Calculator_t0
from alphagen_ocean.callbacks import CustomCallback
from alphagen_ocean.stock_data import ArgData
from utils import SELECTED_CODES

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
args = ap.ArgumentParser()
args.add_argument("--gpu", "-g", type=int, default=1)
args.add_argument("--name", "-n", type=str, default="ret1d")
args.add_argument("--kwargs", "-k", type=str, default="None")
args = args.parse_args()


DEVICE_MODEL = torch.device(f"cuda:{args.gpu}")
# DEVICE_CALC = torch.device(f"cuda:{args.gpu}")
DEVICE_CALC = torch.device("cpu")


def main(
    seed: int = 0,
    instruments: str = "all",
    pool_capacity: int = 10,
    steps: int = 200_000,
):
    reseed_everything(seed)
    calculator_train = Calculator_t0(
        codes=SELECTED_CODES,
        start=20210101,
        end=20211231,
        max_backtrack_ticks=HORIZON_BACK,
        max_future_ticks=0,
        device=DEVICE_CALC,
    )
    calculator_valid = Calculator_t0(
        codes=SELECTED_CODES,
        start=20220101,
        end=20220331,
        max_backtrack_ticks=HORIZON_BACK,
        max_future_ticks=0,
        device=DEVICE_CALC,
    )
    calculator_test = Calculator_t0(
        codes=SELECTED_CODES,
        start=20220331,
        end=20221231,
        max_backtrack_ticks=HORIZON_BACK,
        max_future_ticks=0,
        device=DEVICE_CALC,
    )

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
        save_path="./t0_results",
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
                d_model=128,  # init 128
                dropout=0.1,
                device=DEVICE_MODEL,
            ),
        ),
        gamma=1.0,
        ent_coef=0.1,  # NOTE 1e-2
        batch_size=512,  # 512
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
        steps=100_000,
        # steps=80_000,
        # steps=250_000,
    )
