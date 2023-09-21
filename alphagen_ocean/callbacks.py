import json
import os
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback

from alphagen.config import *
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore


class FixedSizeContainer:
    def __init__(self, size=3):
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
        self._continue = self.earlystop.add(ic_test)
        if not self._continue:
            print("Early stop!".center(60, "="))
        else:
            self.show_pool_state()
            self.save_checkpoint()
        print(f"{self.timestamp}".center(60, "="))

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
