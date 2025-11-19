import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리 접근 가능하도록 설정

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from mid.two_layer_net import TwoLayerNet

# -----------------------------
# 1. 데이터 읽기 및 전처리
# -----------------------------
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 입력 차원 축소 (28×28 → 7×7) : 문제 조건
x_train = x_train[:, ::4]
x_test = x_test[:, ::4]

train_loss_list = []

# -----------------------------
# 2. 하이퍼파라미터 설정 (문제 조건)
# -----------------------------
iters_num = 50         # 반복 횟수
train_size = x_train.shape[0]
batch_size = 80        # 미니배치 크기
learning_rate = 0.1

# 입력 크기를 자동으로 맞춤 (784 → 196)
network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=10)

# -----------------------------
# 3. 학습 루프
# -----------------------------
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산 (빠른 버전)
    grad = network.gradient(x_batch, t_batch)
    # 느린 버전 (확인용)
    # grad = network.numerical_gradient(x_batch, t_batch)

    # 매개변수 갱신 (경사하강법)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 손실값 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# -----------------------------
# 4. 손실 곡선 출력
# -----------------------------
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Q5: Mini-batch Learning (iters=50, batch=80, input=7x7)")
plt.ylim(0, 3)     # 손실값 범위를 작게 (가시성 향상)
plt.xlim(0, 50)    # 반복 횟수 50회에 맞춤
plt.grid(True)
plt.show()