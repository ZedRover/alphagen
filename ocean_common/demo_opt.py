from typing import Union

import torch
from torch import Tensor

U_INSTR_NUM = 6000  # Define your constant here


class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int:
        ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]:
        ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = (
            operand if isinstance(operand, Expression) else Constant(operand)
        )

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return UnaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor:
        ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self):
        return self._operand.is_featured


class Neg(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return -operand


class CrsRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        if operand.shape[0] % U_INSTR_NUM != 0:
            return torch.tensor(float("nan"))

        operand = operand.round(6).reshape(-1, U_INSTR_NUM)
        validnum = torch.isfinite(operand).sum(axis=1, keepdims=True)
        return (
            torch.argsort(torch.argsort(operand, axis=1), axis=1) - (validnum + 1) / 2
        ).reshape(
            -1,
        )


class BiasCrsRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        if operand.shape[0] % U_INSTR_NUM != 0:
            return torch.tensor(float("nan"))

        operand = operand.round(6).reshape(-1, U_INSTR_NUM)
        return torch.argsort(torch.argsort(operand, axis=1), axis=1).reshape(
            -1,
        )


class NormCrsRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        if operand.shape[0] % U_INSTR_NUM != 0:
            return torch.tensor(float("nan"))

        operand = operand.round(6).reshape(-1, U_INSTR_NUM)
        validnum = torch.isfinite(operand).sum(axis=1, keepdims=True)
        return (
            torch.argsort(torch.argsort(operand, axis=1), axis=1) / validnum - 0.5
        ).reshape(
            -1,
        )
