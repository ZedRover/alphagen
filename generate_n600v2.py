import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from alphagen_ocean.calculator_t0 import Calculator_t0
from alphagen.utils.pytorch_utils import standardize_tensor
import numpy as np
import pandas as pd
import torch as th
import os
import json
from utils import *
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import SharedArray as sa

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def load_json_formulas(sigs_dir):
    formulas = {}
    for path in sigs_dir:
        with open(path, "r") as f:
            alpha = json.load(f)
            formulas[path] = alpha
    return formulas


def process_stock(code, formulas, start_time, end_time, max_backtrack_days, device=1):
    try:
        calculator = Calculator_t0(
            codes=[code],
            start=start_time,
            end=end_time,
            max_backtrack_ticks=max_backtrack_days,
            max_future_ticks=0,
            device=th.device("cpu"),
        )
        data_dict = calculator.dat_dict
        data = data_dict[code]
        results = []
        done_forms = []
        error_forms = []
        device = th.device(f"cuda:{device}")
        data._data = data._data.to(device)
        data.device = device
        for path, alpha in formulas.items():
            try:
                with th.no_grad():
                    factors = [
                        (
                            Feature(getattr(FeatureType, expr[1:])).evaluate(data)
                            if expr[0] == "$"
                            else formula_to_expression(expr).evaluate(data)
                        )
                        for expr in alpha["exprs"]
                    ]
                    weights = th.tensor(alpha["weights"], device=device)
                    factor_value = sum(f * w for f, w in zip(factors, weights))
                    factor_value = standardize_tensor(factor_value)
                    padding = th.zeros(max_backtrack_days, 1, device=device)
                    results.append(th.concat([padding, factor_value], dim=0).cpu())
                    done_forms.append(path)
            except Exception as e:
                error_forms.append(path)
                logging.error(f"Error processing formula {path} for stock {code}: {e}")
        del data._data  # 明确删除变量引用，帮助垃圾收集器释放内存
        th.cuda.empty_cache()  # 清空 CUDA 缓存
        signal = th.stack(results, dim=1).squeeze().numpy()
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        # try:
        #     fp = sa.create(f"X_44_{code}", shape=signal.shape, dtype=np.float32)
        # except:
        #     fp = sa.attach(f"X_44_{code}")
        # fp[:] = signal
        np.save(f"/mnt/disk2/factors_may/{code}.npy", signal)
        # np.save(f"/mnt/nas/data/WY/factor_0527/stkCode_{code}.npy", signal)

        logging.info(f"Success processing stock {code}")
    except Exception as e:
        logging.error(f"Error processing stock {code}: {e}")


if __name__ == "__main__":
    start_time = 20210101
    end_time = 20221231
    max_backtrack_days = 200

    sigs_dir = glob("data/May23/*.json")
    sigs_dir = sorted(sigs_dir, key=os.path.getmtime)
    column_names = [os.path.splitext(os.path.basename(path))[0] for path in sigs_dir]
    # 保存列名到文本文件
    with open("/mnt/nas/data/WY/factor_0527/column_names.txt", "w") as file:
        file.write("\n".join(column_names))
    formulas = load_json_formulas(sigs_dir)

    start_total_time = time.time()
    code_list = SELECTED_CODES
    n = len(code_list)

    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                process_stock,
                code_list,
                [formulas] * n,
                [start_time] * n,
                [end_time] * n,
                [max_backtrack_days] * n,
                [1] * n,
            )
        )
    end_total_time = time.time()

    total_execution_time = end_total_time - start_total_time
    avg_execution_time = total_execution_time / n

    logging.info(f"Total execution time: {total_execution_time:.2f} seconds")
    logging.info(f"Average execution time per stock: {avg_execution_time:.2f} seconds")
