import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from mid.two_layer_net import TwoLayerNet

# -----------------------------
# 0. 유틸: 28x28 -> 14x14 다운샘플 (행/열 2칸 간격)
# -----------------------------
def downsample_28_to_14(x_flat):
    """
    x_flat: (N, 784), normalize=True 에서 받은 평탄화 데이터
    return: (N, 196)  == 14x14 평탄화
    """
    N = x_flat.shape[0]
    imgs = x_flat.reshape(N, 1, 28, 28)       # (N, C=1, H, W)
    imgs_14 = imgs[:, :, ::2, ::2]            # H,W를 각각 2칸 간격으로 샘플링 -> (N,1,14,14)
    return imgs_14.reshape(N, 196)

# -----------------------------
# 1. 데이터 읽기 및 전처리
# -----------------------------
# one_hot_label=True: 두레이 책(TwoLayerNet) 인터페이스와 호환
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# (중요) 28x28 -> 14x14 다운샘플링 후 평탄화(196)
x_train = downsample_28_to_14(x_train)
x_test  = downsample_28_to_14(x_test)

# -----------------------------
# 2. 하이퍼파라미터 (문제 조건)
# -----------------------------
iters_num = 3000              # 총 반복 횟수
train_size = x_train.shape[0]
batch_size = 100              # 미니배치 크기
learning_rate = 0.1
hidden_size = 2               # 은닉층 노드 수 (조건)

# 모델: 입력 196, 은닉 2, 출력 10
network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=hidden_size, output_size=10)

# 로그 저장소
train_loss_list = []
train_acc_list = []
test_acc_list = []

# -----------------------------
# 3. 학습 루프
# -----------------------------
for i in range(1, iters_num + 1):
    # 미니배치 샘플링
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기(오차역전파)
    grad = network.gradient(x_batch, t_batch)

    # 파라미터 업데이트
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 손실 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 매 20회마다 정확도 측정 및 출력
    if i % 20 == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"[iter {i:4d}] loss={loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

# -----------------------------
# 4. 그래프 출력
# -----------------------------
# (a) 손실 곡선: 그림 4-11
plt.figure()
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Q5: Loss curve (input=14x14, hidden=2, batch=100, iters=3000)")
plt.grid(True)

# (b) (옵션) 정확도 곡선: 그림 4-12
#      x축은 20-스텝마다의 측정 지점
plt.figure()
x_ticks = np.arange(20, iters_num + 1, 20)
plt.plot(x_ticks, train_acc_list, label="train acc")
plt.plot(x_ticks, test_acc_list, label="test acc")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Q5: Accuracy (every 20 iters)")
plt.legend()
plt.grid(True)

plt.show()