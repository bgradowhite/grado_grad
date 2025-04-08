import os, pickle,gzip, importlib
import grado_grad
from grado_grad.engine import Tensor
import numpy as np



def unpickle(file_path):
        with open(file_path, 'rb') as file:
            loaded_data= pickle.load(file, encoding='latin1')
        return loaded_data
def gz_unpickle(file_path):
        with gzip.open(file_path, 'rb') as file:
            loaded_data= pickle.load(file, encoding='latin1')
        return loaded_data

def get_mnist():
    """Get MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    Training data: tuple (50k images, 28*28=74 len) and (50k,1) labels
    val and test are image/label tuples with 10k entries
    """

    grado_grad_pth = os.path.dirname(grado_grad.__file__)
    data_pth = grado_grad_pth + '/data/MNIST/mnist.pkl.gz'
    data  = gz_unpickle(data_pth)
    data = [(Tensor(X),Tensor(Y, requires_grad=False)) for (X,Y) in data]
    return tuple(data)

