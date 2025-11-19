import numpy as np 

def function_f(x):
    n = 1+2*x[1]+x[0]*x[1]*x[2]+np.power(x[2],3.0)
    return n

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원

    return grad

x = np.array([0.0, 1.0, 2.0])
print(numerical_gradient(function_f, x))