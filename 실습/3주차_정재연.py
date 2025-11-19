import numpy as np

# 입력 벡터 (x1=2, x2=1, x3=-1)
x = np.array([2, 1, -1])

# 가중치 행렬 (문제에서 주어진 값)
W = np.array([
    [1, 0, 3],   # y1에 대한 가중치
    [0, 2, 1]    # y2에 대한 가중치
])

# 출력 계산 (행렬 곱)
y = W @ x

# 결과 출력
print("y1 =", y[0])
print("y2 =", y[1])
