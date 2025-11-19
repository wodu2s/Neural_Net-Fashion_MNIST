# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from data.fashion import mnist_reader
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0. Fashion MNIST 데이터 읽기==========
x_train, t_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, t_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# 1. 하이퍼파라미터 설정==========
iters_num = 10000      # 반복 횟수
train_size = x_train.shape[0]
batch_size = 128       # 미니배치 크기
learning_rate = 0.01   # 학습률 (AdaGrad는 비교적 높은 학습률에서 시작해도 괜찮습니다)

train_loss_list = [] # 학습 손실 기록
train_acc_list = []  # 학습 정확도 기록
test_acc_list = []   # 테스트 정확도 기록

# 1 에폭당 반복 수 계산
iter_per_epoch = max(train_size / batch_size, 1)

# 2. 3층 신경망 및 옵티마이저 생성==========
# 입력 크기: 784, 은닉층: 100개 뉴런, 출력 크기: 10
network = MultiLayerNet(input_size=784, hidden_size_list=[100], output_size=10)
# 옵티마이저로 AdaGrad 사용
optimizer = AdaGrad(lr=learning_rate)

# 3. 학습 수행==========
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grads = network.gradient(x_batch, t_batch)
    
    # 옵티마이저를 이용해 매개변수 갱신
    optimizer.update(network.params, grads)
    
    # 손실 함수 값 계산 및 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1 에폭마다 정확도 계산 및 출력
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch {int(i/iter_per_epoch)}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Loss = {loss:.4f}")

# 4. 손실 및 정확도 곡선 시각화==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Training Progress with AdaGrad')

# 손실 곡선 (첫 번째 subplot)
# smooth_curve의 결과 배열 길이가 입력보다 길 수 있으므로 슬라이싱으로 길이를 맞춥니다.
smoothed_loss = smooth_curve(train_loss_list)[:len(train_loss_list)]
ax1.plot(np.arange(len(train_loss_list)), smoothed_loss)
ax1.set_title("Training Loss Curve")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.grid(True)

# 정확도 곡선 (두 번째 subplot)
x_epochs = np.arange(len(train_acc_list))
ax2.plot(x_epochs, train_acc_list, marker='o', label='Train Accuracy', markevery=1)
ax2.plot(x_epochs, test_acc_list, marker='s', label='Test Accuracy', markevery=1)
ax2.set_title("Training and Test Accuracy per Epoch")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.0)
ax2.legend(loc='lower right')
ax2.grid(True)

# 레이아웃 조정 및 그래프 출력
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
