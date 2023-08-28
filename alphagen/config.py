from alphagen.data.expression import *
from alphagen.data.expression_ocean import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 1024  # 256

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
OPERATORS += OCOperators

DELTA_TIMES = [5, 8, 16, 20, 32, 40, 60]

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
