import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from mid.two_layer_net import TwoLayerNet

# -----------------------------
# 데이터 로드 및 전처리
# -----------------------------
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:, ::4]
x_test  = x_test[:, ::4]

# -----------------------------
# 네트워크 구성
# -----------------------------
network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=10)

# -----------------------------
# 하이퍼파라미터
# -----------------------------
epochs = 20             # epoch 기준으로 반복 (명시적)
batch_size = 80
learning_rate = 0.1

train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

# -----------------------------
# 기록 변수
# -----------------------------
train_loss_list = []
train_acc_list = []
test_acc_list = []

# -----------------------------
# 학습 루프 (epoch 기준)
# -----------------------------
for epoch in range(epochs):
    for i in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    # 매 epoch마다 정확도 평가
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# -----------------------------
# 시각화
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_acc_list, 'o-', label='train acc')
plt.plot(range(1, epochs+1), test_acc_list, 's--', label='test acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Q7: Accuracy Curve (batch=80, input=7x7)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
