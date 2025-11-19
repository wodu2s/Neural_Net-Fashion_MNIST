# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from common.layers import Affine, Relu, SoftmaxWithLoss
from data.fashion import mnist_reader
from common.util import smooth_curve
from common.optimizer import Adam
from common.gradient import numerical_gradient 


class FiveLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, weight_init_std=0.01):
        self.params = {}
        
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, hidden_size4)
        self.params['b4'] = np.zeros(hidden_size4)
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size4, output_size)
        self.params['b5'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu() 
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout) 
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db
        
        return grads

# ==============================================================================
# 학습 실행 부분
# ==============================================================================

x_train, t_train = mnist_reader.load_mnist("data/fashion", kind='train')
x_test, t_test = mnist_reader.load_mnist("data/fashion", kind='t10k')

input_size = x_train.shape[1]
output_size = 10 
train_size = x_train.shape[0]

batch_size = 128
max_iterations = 5000
learning_rate = 0.001

hidden_size1 = 128
hidden_size2 = 128
hidden_size3 = 128
hidden_size4 = 128

network = FiveLayerNet(
    input_size=input_size,
    hidden_size1=hidden_size1,
    hidden_size2=hidden_size2,
    hidden_size3=hidden_size3,
    hidden_size4=hidden_size4,
    output_size=output_size,
    weight_init_std=0.01 
)
optimizer = Adam(lr=learning_rate)

iter_per_epoch = max(1, int(train_size / batch_size))
loss_list = []
train_acc_list = []
test_acc_list = []

print("--- 5층 신경망 학습 시작 ---")

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    
    optimizer.update(network.params, grads) 
    
    loss = network.loss(x_batch, t_batch)
    loss_list.append(loss)

    if i % iter_per_epoch == 0:
        epoch = int(i / iter_per_epoch)
        
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}")

print("\n--- 학습 결과 시각화 ---")
x = np.arange(len(train_acc_list))
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, train_acc_list, label='Train Acc')
plt.plot(x, test_acc_list, label='Test Acc', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.title("Accuracy over Epochs")

plt.subplot(1, 2, 2)
plt.plot(np.arange(max_iterations), smooth_curve(loss_list)[:max_iterations], label='Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Iterations (Smoothed)")

plt.tight_layout()
plt.show()