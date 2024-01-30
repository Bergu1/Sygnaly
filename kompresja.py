import numpy as np
from skimage import io, color
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def fft_forward(variables: NDArray) -> NDArray:
    return np.fft.fft2(variables)


def fft_backward(variables: NDArray) -> NDArray:
    return np.abs(np.fft.ifft2(variables))


def compress(image: NDArray, keep: float) -> NDArray:
    f_transform = fft_forward(image)
    sorted_f = np.sort(np.abs(f_transform.reshape(-1)))

    threshold = sorted_f[int(np.floor((1 - keep) * len(sorted_f)))]
    indices = np.abs(f_transform) > threshold
    f_transform *= indices
    return fft_backward(f_transform)


def compress_with_color(image: NDArray, keep: float) -> NDArray:
    compressed_image = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        compressed_image[..., channel] = compress(image[..., channel], keep)
    return compressed_image


color_image_path = 'lis.jpg'
color_image = io.imread(color_image_path)

gray_image = color.rgb2gray(color_image)

keep_ratio = 0.01
compressed_color_image = compress_with_color(color_image, keep_ratio)
compressed_gray_image = compress(gray_image, keep_ratio)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(color_image)
plt.title("Original Color Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.clip(compressed_color_image.astype(int), 0, 255))
plt.title("Compressed Color Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(compressed_gray_image, cmap='gray')
plt.title("Compressed Grayscale Image")
plt.axis('off')

plt.show()
