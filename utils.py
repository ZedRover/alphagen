from alphagen.data.tree import ExpressionBuilder
from alphagen.data.tokens import *
from alphagen_ocean.stock_data_ import FeatureType

from alphagen.data.expression import *
from alphagen.data.expression_ocean import *
from alphagen.config import OPERATORS
from ocean_common.feature_list import FEATURES


from alphagen.data.tree import ExpressionBuilder
from alphagen.data.tokens import *
from typing import List, Union


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


if __name__ == "__main__":
    formula = "Add(Sub($qnet_incr_cash_cash_equ_ttm,$qamount),Mul($qoper_rev_lyr,$qintra_midprices_5_std))"
    expression = formula_to_expression(formula)
    print(expression)
