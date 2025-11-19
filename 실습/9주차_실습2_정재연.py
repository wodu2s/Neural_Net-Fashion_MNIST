from __future__ import annotations
from typing import Tuple
import numpy as np


class Relu:
    def __init__(self) -> None:
        self.mask: np.ndarray | None = None  # x<=0 위치 저장

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x와 동일 shape의 bool mask 저장
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # ReLU 미분: x>0 이면 1, 아니면 0 -> mask로 차단
        dx = dout.copy()
        dx[self.mask] = 0  # type: ignore[index]
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out: np.ndarray | None = None  # forward의 출력 y 저장

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # σ'(x) = σ(x)*(1-σ(x)); dx = dout * σ'(x)
        y = self.out  # type: ignore[assignment]
        dx = dout * y * (1.0 - y)  # type: ignore[operator]
        return dx


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        assert W.ndim == 2 and b.ndim == 1 and W.shape[1] == b.shape[0]
        self.W: np.ndarray = W
        self.b: np.ndarray = b
        self.x: np.ndarray | None = None

        # 역전파에서 채워지는 그라디언트
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (N, in_dim) 또는 (in_dim,) 가능 -> (N, out_dim)
        self.x = x
        out = x @ self.W + self.b  # 브로드캐스팅으로 행별 더해짐
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.x  # type: ignore[assignment]
        self.dW = x.T @ dout  # type: ignore[operator]
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.y: np.ndarray | None = None  # softmax 결과
        self.t: np.ndarray | None = None  # 정답(one-hot or class index)
        self.loss: float | None = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # 안정적 softmax (배치 지원)
        if x.ndim == 2:
            x_shift = x - x.max(axis=1, keepdims=True)
            exp = np.exp(x_shift)
            return exp / exp.sum(axis=1, keepdims=True)
        x_shift = x - x.max()
        exp = np.exp(x_shift)
        return exp / exp.sum()

    @staticmethod
    def _cross_entropy(y: np.ndarray, t: np.ndarray) -> float:
        eps = 1e-12
        if t.ndim == 1 or t.shape == y.shape[:-1]:
            # one-hot
            return float(-np.sum(t * np.log(y + eps)) / y.shape[0])
        batch = y.shape[0]
        return float(-np.sum(np.log(y[np.arange(batch), t] + eps)) / batch)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        y = self._softmax(x)
        self.y = y
        if t.ndim == y.ndim:
            loss = float(-np.sum(t * np.log(y + 1e-12)) / y.shape[0])
        else:
            batch = y.shape[0]
            loss = float(-np.sum(np.log(y[np.arange(batch), t] + 1e-12)) / batch)
        self.loss = loss
        return loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        y = self.y  
        t = self.t  
        batch = y.shape[0] if y.ndim == 2 else 1
        if t.ndim == y.ndim:
            dx = (y - t) / batch
        else:
            dx = y.copy()
            dx[np.arange(batch), t] -= 1
            dx /= batch
        return dx * dout
