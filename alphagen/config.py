from alphagen.data.expression import *
from alphagen.data.expression_ocean import *
from alphagen.data.expression_wq import *

MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 512  # 256

OPERATORS = [
    # Unary
    Abs,
    Sign,
    Log,
    # Binary
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    # Rolling
    Ref,
    Mean,
    Sum,
    Std,
    Var,
    Skew,
    Kurt,
    Max,
    Min,
    Med,
    Mad,
    Rank,
    # Delta,
    WMA,
    EMA,
    # Pair rolling
    Cov,
    Corr,
]
OPERATORS += Operators_oc + Operators_wq

# DELTA_TIMES = [5, 8, 16, 20, 32, 40, 60, 70, 80, 90, 99]
DELTA_TIMES = np.arange(1, 20 * 8, step=32).tolist()

CONSTANTS = [
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
]

REWARD_PER_STEP = 0.0
HORIZON_BACK = max(DELTA_TIMES) + 1
# 320 for ret1d | checkpoints/20230912180154_ret1d_lexpr09_lopt58_10_4798 -> checkpoints/20230918131205_ret1d_lexpr09_lopt58_10_3315
# DELTA_TIMES = np.arange(1, 20 * 16, step=16).tolist()
