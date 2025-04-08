import grado_grad.nn as nn
from grado_grad.engine import Tensor, cross_entropy
import grado_grad.optim as ggoptim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



class MLP(nn.Module):
    def __init__(self,learning_rate=1e-3,
                 dtype=None, device =None):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #init vars
        self.learning_rate = learning_rate
        self.linear1 = nn.Linear(28 * 28, 64)
        self.linear2 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 10)


        self.optimizer = ggoptim.SGD(
            self.parameters(),
            self.learning_rate
        )
        
    def forward(self, input):
        x = self.relu1(self.linear1(input))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x


    def get_error(self, data, batch_size):
        X,Y = data
        test_loss, cor = 0.0,0.0
        size = Y.shape[0]
        n_b = size//batch_size
        with nn.NoGrad():
            for i in range(n_b):
                X_b = X[i*batch_size:(i+1)*batch_size]
                Y_b = Y[i*batch_size:(i+1)*batch_size]
                pred = self(X_b)
                test_loss += (cross_entropy(pred,Y_b).sum(axes=0,keepdims=False)/(pred.shape[0])).data
                cor+=sum([pred[i].argmax(0).data ==Y_b[i].data for i in range(batch_size)])
            test_loss/=n_b
            cor/=size
            print(f"Test Error: \n Accuracy: {(100*cor):>0.1f}%, Avg loss: {test_loss:>8f}\n")
            return test_loss, cor



    def update(self,input, label):
        self.optimizer.zero_grad()
        logits = self(input)
        loss = cross_entropy(logits, label).sum(axes=0,keepdims=False)/(logits.shape[0])
        loss.backward()
        pb, pa = [],[]
        for p in self.optimizer.params:
            pb.append(np.copy(p.data))
        self.optimizer.step()
        for p in self.optimizer.params:
            pa.append(np.copy(p.data))
        #print("here",[100000000*(pb[i]-pa[i]) for i in range(len(pb))])
        return {
            'Training Loss': loss.data,
        }
    

    def train_model(self, data, epochs:int, batch_size:int, device, dtype, lr_schedule=False):
        train_data, val_data, test_data = data
        X,Y = train_data
        n = Y.shape[0]
        train_loss, val_acc = [],[]
        n_b = n//batch_size
        bar = tqdm(range(epochs),total=epochs)
        for _ in bar:
            #shuffle data and batch
            for i in range(n_b):
                X_b = X[i*batch_size:(i+1)*batch_size]
                Y_b = Y[i*batch_size:(i+1)*batch_size]
                info = self.update(X_b, Y_b)
                train_loss.append(info['Training Loss'])
                bar.set_description(f'Training Loss {info['Training Loss']:.2f}')
            val_acc.append(self.get_error(val_data, batch_size))

       
        test_acc = self.get_error(test_data, batch_size)
  
        
        plt.plot(train_loss, color='blue')
        plt.xlabel('training examples')
        plt.ylabel('loss')
        plt.show()
