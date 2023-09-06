from alphagen.data.expression import *
from alphagen.data.expression_ocean import *
from alphagen.data.expression_wq import *

MAX_EXPR_LENGTH = 9
MAX_EPISODE_LENGTH = 256  # 256

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

DELTA_TIMES = [5, 8, 16, 20, 32, 40, 60, 70, 80, 90, 99]

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
