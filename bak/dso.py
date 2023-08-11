import numpy as np

from alphagen.data.expression import *
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *


funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(["open", "close", "high", "low", "volume", "vwap"]):
    funcs[f"x{i+1}"] = Token(
        name=feature, arity=0, complexity=1, function=None, input_var=i
    )
for v in [
    -30.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    -0.01,
    0.01,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
]:
    funcs[f"Constant({v})"] = HardCodedConstant(name=f"Constant({v})", value=v)


instruments = "csi300"
import sys

seed = int(sys.argv[1])
reseed_everything(seed)

cache = {}
device = torch.device("cuda:0")
data = StockData(instruments, "2009-01-01", "2018-12-31", device=device)
data_valid = StockData(instruments, "2019-01-01", "2019-12-31", device=device)
data_test = StockData(instruments, "2020-01-01", "2021-12-31", device=device)


if __name__ == "__main__":
    X = np.array([["open_", "close", "high", "low", "volume", "vwap"]])
    y = np.array([[1]])
    functions.function_map = funcs

    pool = AlphaPool(capacity=10, stock_data=data, target=target, ic_lower_bound=None)

    class Ev:
        def __init__(self, pool):
            self.cnt = 0
            self.pool = pool
            self.results = {}

        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ret = self.pool.try_new_expr(expr)
            except OutOfDataRangeError:
                ret = -1.0
            finally:
                self.cnt += 1
                if self.cnt % 100 == 0:
                    test_ic = pool.test_ensemble(data_test, target)[0]
                    self.results[self.cnt] = test_ic
                    print(self.cnt, test_ic)
                return ret

    ev = Ev(pool)

    config = dict(
        task=dict(
            task_type="regression",
            function_set=list(funcs.keys()),
            metric="alphagen",
            metric_params=[lambda key: ev.alpha_ev_fn(key)],
        ),
        training={"n_samples": 20000, "batch_size": 128, "epsilon": 0.05},
        prior={"length": {"min_": 2, "max_": 20, "on": True}},
    )

    # Create the model
    model = DeepSymbolicRegressor(config=config)
    model.fit(X, y)

    print(ev.results)
