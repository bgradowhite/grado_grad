import random, math
import numpy as np
from grado_grad.engine import Tensor, arange


grad_tracking_enabled = True

class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        return super().__init__(tensor.data, requires_grad=requires_grad)

class Module:

    def __init__(self):
        self.training= True
        self._modules = {}
        self._parameters = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def modules(self): 
        yield from self._modules.values()

    def parameters(self):
        yield from self._parameters.values()
        for mod in self.modules():
            yield from mod.parameters()

    def __setattr__(self, key, val) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        """
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        super().__setattr__(key, val)

class Linear(Module):
    """
    Applies a linear transformation to the input. 
    Args: 
    in_features: dimension of the input
    out_features: dimension of the output
    """
    def __init__(self, in_features:int, out_features:int, bias = True, device=None, dtype=None):
        super().__init__()
        k = 1/math.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(np.random.uniform(low=-k,high=k,size = (in_features,out_features))))
        self.bias = Parameter(Tensor(np.zeros(out_features))) if bias else None
        
    def forward(self,input):
        out  = input.matmul(self.weight)
        if self.bias is not None: 
            out = out + self.bias
        return out
    

class Dropout(Module):
    """
    Applies dropout with probability p. 
    Args: 
    p: probability
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self,input):
        if not self.training or self.p==0:
            return input
        mask = Tensor(np.uniform(low=0,high=1,shape=input.shape)>self.p)
        return (1/(1-self.p))*mask*input
    
class ReLU(Module):
    def forward(self, x):
        return x.relu()

class NoGrad: 
    was_enabled: bool

    def __enter__(self):
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False
    
    def __exit__(self, type, value, traceback):
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled

