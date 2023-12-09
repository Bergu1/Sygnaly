import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from matplotlib import pyplot as plt

image = io.imread("obrazek2.jpg")
image = rgb2gray(image)
image = resize(image, output_shape=(500, 500))


def max_pooling(im, pool_size=2):
    new_x = im.shape[0] // pool_size
    new_y = im.shape[1] // pool_size
    result = np.zeros((new_x, new_y))
    print(result)
    for x in range(new_x):
        for y in range(new_y):
            result[x, y] = np.max(im[x * pool_size:(x + 1) * pool_size, y * pool_size:(y + 1) * pool_size])

    return result


downsized_max_pooling = max_pooling(image)


def conv1d_interpolate(x_measure, y_measure, x_interpolate, kernel):
    width = x_measure[1] - x_measure[0]  # store period between samples
    kernels = [kernel(x_interpolate, x0=offset, w=width) for offset in x_measure]

    return y_measure @ kernels


def kernel3(x, x0: float, w: float):
    x = x - x0
    x = x / w
    return np.array([1 - np.abs(xi) if -1 <= xi <= 1 else 0 for xi in x])


x = np.linspace(0, 1, len(downsized_max_pooling[0, :]))
x_interpolate = np.linspace(0, 1, 2 * len(downsized_max_pooling[0, :]))
interpolated = conv1d_interpolate(x_measure=x, y_measure=downsized_max_pooling[0, :], x_interpolate=x_interpolate,
                                  kernel=kernel3)
result = np.zeros([2 * downsized_max_pooling.shape[0], downsized_max_pooling.shape[1]])

for row in range(downsized_max_pooling.shape[1]):
    result[:, row] = conv1d_interpolate(x_measure=x, y_measure=downsized_max_pooling[row, :],
                                        x_interpolate=x_interpolate, kernel=kernel3)
x = np.linspace(0, 1, len(downsized_max_pooling[0, :]))
io.imshow(result.T)
