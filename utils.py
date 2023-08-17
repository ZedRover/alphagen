import json
import time
from typing import List, Union

from alphagen.config import DEVICE_DATA, OPERATORS
from alphagen.data.expression import *
from alphagen.data.expression_ocean import *
from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_ocean.stock_data_ import FeatureType, StockData


def tokenize_formula(formula: str) -> List[str]:
    tokens = []
    token = ""
    i = 0
    while i < len(formula):
        char = formula[i]
        if char == "(" or char == "," or char == ")":
            if token:
                tokens.append(token.strip())
                token = ""
            tokens.append(char)
            i += 1
        elif char == "$":
            token += char
            i += 1
            while i < len(formula) and formula[i].isalnum():
                token += formula[i]
                i += 1
        elif formula[i : i + 8] == "Constant":
            i += 8
            const_val = ""
            while formula[i] != ")":
                const_val += formula[i]
                i += 1
            token = "Constant" + const_val
            tokens.append(token.strip())
            token = ""
            i += 1  # skip the closing parenthesis
        else:
            token += char
            i += 1
    if token:
        tokens.append(token.strip())
    return tokens


def infix_to_rpn(
    tokens: List[str], operators: List[Type[Operator]] = OPERATORS
) -> List[str]:
    output = []
    stack = []

    def is_operator(token: str) -> bool:
        return any(op.__name__ == token for op in operators)

    for token in tokens:
        if (
            token.startswith("$")
            or token.isdigit()
            or (token[0] == "-" and token[1:].isdigit())
            or token.startswith("Constant")
        ):
            output.append(token)
        elif is_operator(token):
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ",":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()  # remove the open parenthesis
        else:
            raise ValueError(f"Unknown token: {token}")

    while stack:
        output.append(stack.pop())
    return output


def formula_to_expression(formula: str) -> Expression:
    # 将公式转换为token列表
    tokens = tokenize_formula(formula)

    # 将token列表转换为RPN
    rpn_tokens = infix_to_rpn(tokens)

    # 使用ExpressionBuilder构建表达式
    builder = ExpressionBuilder()
    for token in rpn_tokens:
        if token.startswith("$"):
            feature_type = getattr(FeatureType, token[1:])
            builder.add_token(FeatureToken(feature_type))
        elif token.startswith("Constant"):
            value = float(token[9:])
            builder.add_token(ConstantToken(value))
        elif token.isdigit() or (token[0] == "-" and token[1:].isdigit()):
            try:
                builder.add_token(DeltaTimeToken(int(token)))
            except:
                builder.add_token(ConstantToken(int(token)))
        else:
            operator_class = next(
                (op for op in OPERATORS if op.__name__ == token), None
            )
            if operator_class:
                builder.add_token(OperatorToken(operator_class))
            else:
                raise ValueError(f"Unknown operator: {token}")

    if not builder.is_valid():
        raise ValueError("Invalid formula")

    return builder.get_tree()


##########################################################################################


def calc_topk(input: Tensor, target: Tensor, topk: int = 10) -> Tensor:
    idx = ~torch.isnan(target)
    input, target = input[idx], target[idx]
    top_q_indices = torch.topk(
        input, int(len(target) * topk / 100), largest=True
    ).indices
    q_ret = torch.nanmean(target[top_q_indices])
    return q_ret


def batch_topk(yhat, y):
    q90 = []
    q99 = []
    for i in range(len(yhat)):
        yhat_ = yhat[i]
        y_ = y[i]
        q90.append(calc_topk(yhat_, y_, 10).item())
        q99.append(calc_topk(yhat_, y_, 1).item())
    q90 = np.mean(q90)
    q99 = np.mean(q99)
    return round(q90, 5), round(q99, 5)


def backtest_json(
    path: str,
    start_date: int = 20210101,
    end_date: int = 20210601,
):
    device = torch.device("cpu")
    data_test = StockData(start_time=start_date, end_time=end_date, device=device)

    with open(path, "r") as f:
        alpha = json.load(f)

    # 使用列表推导式代替传统的for循环
    factors = [
        Feature(getattr(FeatureType, expr[1:])).evaluate(data_test)
        if expr[0] == "$"
        else formula_to_expression(expr).evaluate(data_test)
        for expr in alpha["exprs"]
    ]

    weights = torch.tensor(alpha["weights"])

    # 使用PyTorch的向量化操作来计算factor_value
    factor_value = sum(f * w for f, w in zip(factors, weights))
    factor_value = normalize_by_day(factor_value)

    return factor_value


if __name__ == "__main__":
    formula = "Cov(Abs(Std($qfree_shares_today,50)),$qvalue_diff_large_trader_act,50)"
    expression = formula_to_expression(formula)
    print(expression)
    data_test = StockData(
        start_time=20210101,
        end_time=20210601,
        device=DEVICE_DATA,
    )
    s = time.time()
    expression.evaluate(data_test)
    e = time.time()
    print("Signal calculation time:", e - s)
