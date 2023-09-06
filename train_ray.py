import random

import ray
import torch

from train_lstm import *

ray.init(num_cpus=160, num_gpus=4)


@ray.remote(num_cpus=10, num_gpus=4)
def remote_main(i):
    print(torch.cuda.is_available())
    main(
        seed=random.randint(0, 999),  # trunk-ignore(ruff/S311)
        instruments=f"satd_lexpr{MAX_EXPR_LENGTH}_lopt{len(OPERATORS)}",
        pool_capacity=10,
        steps=250_000,
    )


results = [remote_main.remote(i) for i in range(3)]

results = ray.get(results)

ray.shutdown()
