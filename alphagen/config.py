from alphagen.data.expression import *
from alphagen.data.expression_ocean import *


DEVICE_MODEL = torch.device("cuda:1")
DEVICE_DATA = torch.device("cpu")
DEVICE_CALC = torch.device("cpu")

MAX_EXPR_LENGTH = 5
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
OPERATORS = OPERATORS
OPERATORS += OCOperators

DELTA_TIMES = [5, 10, 15, 20, 30, 40, 50]

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
