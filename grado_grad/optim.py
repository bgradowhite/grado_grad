import numpy as np
from typing import Iterable
from grado_grad.nn import NoGrad



class SGD:
    def __init__(self, params, lr: float):
        """SGD with no additional features."""
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

    def step(self) -> None:
        with NoGrad():
            for p in self.params:
                p.data  = p.data - p.grad*self.lr
