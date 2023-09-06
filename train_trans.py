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
from alphagen.rl.policy import TransformerSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_ocean.calculator import QLibStockDataCalculator
from alphagen_ocean.stock_data import ArgData

args = ap.ArgumentParser()
args.add_argument("--gpu", "-g", type=int, default=1)
args.add_argument("--name", "-n", type=str, default="satd")
args.add_argument("--kwargs", "-k", type=str, default="None")
args = args.parse_args()


DEVICE_MODEL = torch.device(f"cuda:{args.gpu}")
DEVICE_DATA = torch.device("cpu")
DEVICE_CALC = torch.device("cpu")


class FixedSizeContainer:
    def __init__(self, size=5):
        self.container = [-1] * size
        self.size = size

    def add(self, element):
        self.container.pop(0)
        self.container.append(element)
        return self.check_order()

    def check_order(self):
        for i in range(1, self.size):
            if self.container[i] >= self.container[i - 1]:
                return True
        return False


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        show_freq: int,
        save_path: str,
        valid_calculator: AlphaCalculator,
        test_calculator: AlphaCalculator,
        name_prefix: str = "rl_model",
        timestamp: Optional[str] = None,
        verbose: int = 0,
        patience: int = 5,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        self.earlystop = FixedSizeContainer(size=patience)
        self._continue = True

        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return self._continue

    def _on_rollout_end(self) -> None:
        self.logger.record("timesteps", self.num_timesteps)
        self.logger.record("pool/size", self.pool.size)
        self.logger.record(
            "pool/significant",
            (np.abs(self.pool.weights[: self.pool.size]) > 1e-4).sum(),
        )
        self.logger.record("pool/best_ic_ret", self.pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", self.pool.eval_cnt)
        ic_test, pic_test = self.pool.test_ensemble(self.valid_calculator)
        self.logger.record("test/ic", ic_test)
        self.logger.record("test/pool_ic", pic_test)
        self.save_checkpoint()
        self._continue = self.earlystop.add(ic_test)
        if not self._continue:
            print("Early stop!".center(60, "-"))

    def save_checkpoint(self):
        path = os.path.join(
            self.save_path,
            f"{self.timestamp}_{self.name_prefix}",
            f"{self.num_timesteps}_steps",
        )
        self.model.save(path)  # type: ignore
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state["exprs"])
        print("---------------------------------------------")
        for i in range(n):
            weight = state["weights"][i]
            expr_str = str(state["exprs"][i])
            ic_ret = state["ics_ret"][i]
            print(f"> Alpha #{i}: {weight}, {expr_str}, {ic_ret}")
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print("---------------------------------------------")

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


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
    )
    data_valid = ArgData(
        start_time=20210101,
        end_time=20211231,
        device=DEVICE_DATA,
    )
    data_test = ArgData(
        start_time=20210101,
        end_time=20211231,
        device=DEVICE_DATA,
    )
    print("train days:", data_train.n_days)
    print("  val days:", data_valid.n_days)
    print(" test days:", data_test.n_days)

    calculator_train = QLibStockDataCalculator(data_train, device=DEVICE_CALC)
    calculator_valid = QLibStockDataCalculator(data_valid, device=DEVICE_CALC)
    calculator_test = QLibStockDataCalculator(data_test, device=DEVICE_CALC)

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
            features_extractor_class=TransformerSharedNet,
            features_extractor_kwargs=dict(
                n_encoder_layers=2,
                d_model=512,  # init 128
                n_head=4,
                d_ffn=512 * 2,
                dropout=0.1,
                device=DEVICE_MODEL,
            ),
        ),
        gamma=1.0,
        ent_coef=0.1,  # NOTE 1e-2
        batch_size=512,
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
        seed=random.randint(0, 9999),  # trunk-ignore(ruff/S311)
        instruments=f"{args.name}_attn_lexpr{str(MAX_EXPR_LENGTH).zfill(2)}_lopt{len(OPERATORS)}",
        pool_capacity=10,
        steps=250_000,
    )
