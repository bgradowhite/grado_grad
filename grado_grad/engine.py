import numpy as np
def unbroad(tensor, grd):
    #completely sum over dimensions that were previously nonexistent
    dims_sums = len(grd.shape) - len(tensor.shape)
    grd = grd.sum(axis=tuple(range(dims_sums)))
    #collapse dimensions that were previously one
    dims_sums = tuple([i for i, (t, o) in enumerate(zip(tensor.shape, grd.shape)) if t == 1 and o > 1])
    grd= grd.sum(axis=dims_sums, keepdims=True)
    return grd


def coerce_index(index):
    """Helper function: converts array of tensors to array of numpy arrays."""
    if isinstance(index, tuple) and all(isinstance(i, Tensor) for i in index):
        return tuple([i.data for i in index])
    else:
        return index

class Tensor: 
    def __init__(self, data, _children=(), requires_grad=True):
        self.data = data if isinstance(data,np.ndarray) else np.array(data)
        self.grad = np.zeros_like(data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.shape = (self.data).shape
        self.len = len(self.shape)
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self, gradient = None):
        self.grad = gradient if gradient is not None else np.ones_like(self.data)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self,other))
        def _backward():
            #We can make our backprop through addition 
            # compatible with broadcasting by summing over
            # any broadcasted dimension
            def bkprop_brdcst(tensor):
                grd = out.grad
                grd = unbroad(tensor, grd)
                return grd
            if self.requires_grad:
                self.grad+=bkprop_brdcst(self)
            if other.requires_grad:
                other.grad+= bkprop_brdcst(other)         
        out._backward = _backward
        return out
    
           
    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other))
        def _backward():
            def bkprop_brdcst(tensor,multens):
                grd = out.grad*(multens.data)
                return unbroad(tensor,grd)
            if self.requires_grad:
                self.grad+=bkprop_brdcst(self, other)
            if other.requires_grad:
                other.grad+= bkprop_brdcst(other,self)         
        out._backward = _backward
        return out

    def __getitem__(self, index):
        #For Tensor slicing/indexing
        out = Tensor(self.data[coerce_index(index)], (self,))
        def _backward(): 
            if self.requires_grad:
                np.add.at(self.grad, coerce_index(index), out.grad)
        out._backward = _backward
        return out

    def matmul(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data,other.data), (self,other))
        def _backward():
            def bkprop_brdcst(t1,t2):
                #get derivative
                grd1 = np.einsum("...ij,...kj->...ik",out.grad,t2.data)
                grd2 = np.einsum("...ij,...ik->...kj", out.grad, t1.data)
                #collapse over broadcasted dimensions
                grd1 = unbroad(t1, grd1)
                grd2 = unbroad(t2,grd2)
                return grd1,grd2
            grd1, grd2 = bkprop_brdcst(self, other)
            if self.requires_grad:
                self.grad+= grd1
            if other.requires_grad:
                other.grad+= grd2
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad = False)
        out = Tensor(self.data/other.data, (self,other))
        def _backward():
            def bkprop_brdcst(num,den):
                grdnum = out.grad/(den.data)
                grdden = -out.grad*(num.data/(den.data**2))
                grdnum = unbroad(num,grdnum)
                grdden = unbroad(den, grdden)
                return grdnum, grdden
            grdnum, grdden = bkprop_brdcst(self, other)
            if self.requires_grad:
                self.grad+=grdnum
            if other.requires_grad:
                other.grad+=grdden  
        out._backward = _backward
        return out
    
    
    def reshape(self,shape):
        out = Tensor(self.data.reshape(shape), (self,))
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad.reshape(self.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, other): 
        assert isinstance(other, (int, float)), "Must be raise to an int/float power"
        out = Tensor(self.data**other, (self,))
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad*(other)*self.data**(other-1)
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.data*out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad/self.data
        out._backward = _backward
        return out



    def relu(self):
        out = Tensor((self.data>0)*self.data, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += (self.data>0)*out.grad
        out._backward = _backward
        return out
    
    def sum(self, axes, keepdims): 
        out = Tensor(np.sum(self.data, axis = axes, keepdims=keepdims), (self,))
        def _backward():
            if self.requires_grad:
                if not keepdims:
                    self.grad+=np.expand_dims(out.grad, axes)* np.ones_like(self.data)
                else: 
                    self.grad+=out.grad* np.ones_like(self.data)
        out._backward = _backward
        return out

    def max(self, axes, keepdims):
        out = Tensor(np.max(self.data, axis=axes, keepdims=keepdims), (self,))
        def _backward():
            if self.requires_grad:
                if keepdims: 
                    self.grad+=(self.data==np.max(self.data,axis=axes,keepdims=True))*out.grad
                else: 
                    self.grad+=(self.data==np.max(self.data, axis=axes, keepdims=True))*np.expand_dims(out.grad, axes)
        out._backward = _backward
        return out
    
    def argmax(self, axes=None, keepdims=False):
        out = Tensor(np.argmax(self.data, axis=axes, keepdims=keepdims), requires_grad=False)
        return out
        

    def min(self, axes, keepdims):
        out = Tensor(np.min(self.data, axis=axes, keepdims=keepdims), (self,))
        def _backward():
            if self.requires_grad:
                if keepdims: 
                    self.grad+=(self.data==np.min(self.data))*out.grad
                else: 
                    self.grad+=(self.data==np.min(self.data))*np.expand_dims(out.grad, axes)
        out._backward = _backward
    """
    def log_softmax(input,dim=-1):
        max_val = input.max(dim=dim, keepdim=True).values
        stab_inp = input-max_val
        stab_inp.exp(dim=dim, keepdim=True)
        return stab_inp - torch.log(torch.sum(torch.exp(stab_inp),dim=dim, keepdim=True))

    def cross_entropy(self, other):
        loss = torch.sum(-log_softmax(pred, dim=-1)*oh_t, dim=-1)
    """
    


    def __neg__(self):
        return self*(-1.0)
    
    def __sub__(self,other):
        return self + (-other)
    

    def __radd__(self, other):
        return self + other
    def __rmul__(self,other):
        return self*other
    
    def __rtruediv__(self,other):
        return other*(self**(-1))
    def __rsub__(self,other):
        return other + (-self)
    




def arange(start: int, end: int, step=1):
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def cross_entropy(logits: Tensor, true_labels:Tensor)->Tensor:
    """
    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.
    Return: shape (batch, )
    """
    b = logits.shape[0]
    logprobs = logits - ((logits.exp()).sum(-1, keepdims=True)).log()
    out = -logprobs[arange(0,b), true_labels]
    return out