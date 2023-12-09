import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


def sin(x):
    x_safe = np.where(x == 0, 1e-10, x)
    return np.sin(x_safe)


x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
y = sin(x)
plt.scatter(x, y)
plt.show()


def kernel1(x, x0, w):
    x = x - x0
    x = x / w
    return np.array([1 if 0 <= xi < 1 else 0 for xi in x])


def kernel2(x, x0, w):
    x = x - x0
    x = x / w
    return np.array([1 if -1 / 2 <= xi < 1 / 2 else 0 for xi in x])


# interpolacja z użyciem jądra h3
def kernel3(x, x0, w):
    x = x - x0
    x = x / w
    return np.array([1 - np.abs(xi) if -1 <= xi <= 1 else 0 for xi in x])


width = np.diff(x)[0]
kernels1 = []
kernels2 = []
kernels3 = []
# xk dla jadra h3
xk3 = np.linspace(-2 * np.pi, 2 * np.pi, 50)

for xsample, ysample in zip(x, y):
    k = ysample * kernel3(xk3, x0=xsample, w=width)
    kernels3.append(k)

for xsample, ysample in zip(x, y):
    k = ysample * kernel1(xk3, x0=xsample, w=width)
    kernels1.append(k)

for xsample, ysample in zip(x, y):
    k = ysample * kernel2(xk3, x0=xsample, w=width)
    kernels2.append(k)

kernels1 = np.asarray(kernels1).sum(axis=0)
kernels2 = np.asarray(kernels2).sum(axis=0)
kernels3 = np.asarray(kernels3).sum(axis=0)

plt.plot(xk3, kernels1, color="red", label="1 jądro")
plt.plot(xk3, kernels2, color="black", label="2 jądro")
plt.plot(xk3, kernels3, color="green", label="3 jądro")
plt.scatter(x, y, color="green", label="punkty")
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

print(f"{(mean_squared_error(kernels1, np.sin(xk3))) / np.std(np.sin(xk3)):.4%}")
print(f"{(mean_squared_error(kernels2, np.sin(xk3))) / np.std(np.sin(xk3)):.4%}")
print(f"{(mean_squared_error(kernels3, np.sin(xk3))) / np.std(np.sin(xk3)):.4%}")
