import torch
import numpy as np 
import engine  as gg
#import nn
#mport optim




def comp_np_torch(nparr, torchtens):
    np2pt = torch.from_numpy(nparr)
    assert np2pt.shape==torchtens.shape, "Bad Shape"
    assert torch.allclose(np2pt, torchtens), "Bad Values"

def add_test():
    a = np.random.rand(3,1)
    b = np.random.rand(4)
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b)
    outgg = agg+bgg
    gradient = np.random.rand(*(a+b).shape)
    outgg.backward(gradient)

    apt = torch.from_numpy(a)
    apt.requires_grad=True
    bpt = torch.from_numpy(b)
    bpt.requires_grad = True
    outpt = apt+bpt
    outpt.backward(torch.from_numpy(gradient))

    comp_np_torch(outgg.data,outpt)
    comp_np_torch(agg.grad, apt.grad)
    comp_np_torch(bgg.grad, bpt.grad)


def mul_test():
    a = np.random.rand(3,1)
    b = np.random.rand(1,1,4)
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b)
    outgg = agg*bgg
    gradient = np.random.rand(*(a*b).shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    bpt = torch.from_numpy(b)
    bpt.requires_grad = True
    outpt = apt*bpt
    outpt.backward(torch.from_numpy(gradient))

    comp_np_torch(outgg.data,outpt)
    comp_np_torch(agg.grad, apt.grad)
    comp_np_torch(bgg.grad, bpt.grad)


def matmul_test():
    a = np.random.rand(2,1,4,3)
    b = np.random.rand(2,3,5)
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b)
    outgg = agg.matmul(bgg)
    gradient = np.random.rand(*(a@b).shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    bpt = torch.from_numpy(b)
    bpt.requires_grad = True
    outpt = apt@bpt
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)
    #print(agg.grad, apt.grad)
    comp_np_torch(bgg.data,bpt.data)
    comp_np_torch(bgg.grad, bpt.grad)



def truediv_test():
    a = np.random.rand(2,3,5,1)
    b = np.random.rand(1,5)
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b)
    outgg = agg/bgg
    gradient = np.random.rand(*(a/b).shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    bpt = torch.from_numpy(b)
    bpt.requires_grad = True
    outpt = apt/bpt
    outpt.backward(torch.from_numpy(gradient))

    comp_np_torch(outgg.data,outpt)
    comp_np_torch(agg.grad, apt.grad)
    comp_np_torch(bgg.grad, bpt.grad)

def reshape_test():
    inp_shape = (2,2)
    out_shape = (4,)
    a = np.random.rand(*inp_shape)
    agg = gg.Tensor(a)
    outgg = agg.reshape(out_shape)
    gradient = np.random.rand(*out_shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.reshape(apt,out_shape)
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)

def relu_test():
    a = np.random.rand(2,3,5,7)
    agg = gg.Tensor(a)
    outgg = agg.relu()
    gradient = np.random.rand(*agg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.nn.functional.relu(apt)
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)


def log_test():
    a = np.random.rand(2,3,7)
    agg = gg.Tensor(a)
    outgg = agg.log()
    gradient = np.random.rand(*agg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.log(apt)
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)

def sum_test():
    in_shape = (2,3,7)
    axes = (0,2)
    keepdims = True

    a = np.random.rand(*in_shape)
    agg = gg.Tensor(a)
    outgg = agg.sum(axes,keepdims)
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.sum(apt, axes, keepdim=keepdims)
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)
    


def max_test():
    in_shape = (5,3,4,5)
    axes = 1
    keepdims = True

    a = np.random.rand(*in_shape)
    agg = gg.Tensor(a)
    outgg = agg.max(axes,keepdims)
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)

    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.max(apt, axes, keepdims=keepdims).values
    grad = gradient if isinstance(gradient, float) else torch.from_numpy(gradient)
    outpt.backward(torch.ones_like(outpt)*grad)
 

    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    #(agg.grad, apt.grad)
    comp_np_torch(agg.grad, apt.grad) 
    
   
def get_item_test():
    a = np.random.rand(5,5)
    index = (gg.arange(0,1), gg.Tensor(np.array((0,1))))
    agg = gg.Tensor(a)
    outgg = -agg[index]
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)

    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = -apt[torch.arange(0,1), torch.tensor((0,1))]
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)


def neg_test(): 
    a = np.random.rand(3,1)
    agg = gg.Tensor(a)
    outgg = -agg
    gradient = np.random.rand(*(a).shape)
    outgg.backward(gradient)

    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = -apt
    outpt.backward(torch.from_numpy(gradient))

    comp_np_torch(outgg.data,outpt)
    comp_np_torch(agg.grad, apt.grad)

def cross_entropy_test():
    a = np.random.rand(5,8)
    b = np.array((0,4,2,1,3))
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b, requires_grad=False)
    out1 = gg.cross_entropy(agg,bgg)
    out2 = out1.sum(axes=-1,keepdims=False)
    outgg = out2/(agg.shape[0])
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt =  torch.nn.functional.cross_entropy(apt,torch.from_numpy(b))
    outpt.backward(torch.tensor(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)


def pow_test():
    a = np.random.rand(5,8)
    agg = gg.Tensor(a)
    outgg = agg**(-0.5)
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)

    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = apt**(-0.5)
    outpt.backward(torch.tensor(gradient))

    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)


def exp_test():
    a = np.random.rand(2,3,7)
    agg = gg.Tensor(a)
    outgg = agg.exp()
    gradient = np.random.rand(*agg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad=True
    outpt = torch.exp(apt)
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)

def argmax_test():
    axes = 0
    keepdims = False
    a = np.random.rand(2,3,7)

    agg = gg.Tensor(a)
    out1gg = agg.argmax(axes=axes, keepdims=keepdims)
    outgg = 4.0*agg+2.0*out1gg
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    apt.requires_grad = True
    out1pt = torch.argmax(apt, dim=axes, keepdim=keepdims)
    outpt =  4.0*apt+2.0*out1pt 
 
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)

def sub_test():
    a = np.random.rand(2,3,7)
    b = np.random.rand(1,3,7)
    agg = gg.Tensor(a)
    bgg = gg.Tensor(b)
    out1gg = b/agg
    outgg = 4.0*agg+2.0*out1gg
    gradient = np.random.rand(*outgg.shape)
    outgg.backward(gradient)
    
    apt = torch.from_numpy(a)
    bpt = torch.from_numpy(b)
    apt.requires_grad = True
    bpt.requires_grad = True
    out1pt =  b/apt
    outpt =  4.0*apt+2.0*out1pt 
 
    outpt.backward(torch.from_numpy(gradient))
 
    comp_np_torch(outgg.data,outpt.data)
    comp_np_torch(agg.data,apt.data)
    comp_np_torch(agg.grad, apt.grad)

neg_test()
