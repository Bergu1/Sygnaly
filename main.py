import numpy as np
from skimage import io, color
from scipy.ndimage import convolve
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

image = io.imread("images.jpg")
image_gray = color.rgb2gray(image)
image2 = io.imread("pobrane.png")
image2_gray = color.rgb2gray(image2)


def max_pool(image, kernel, stride: int):
    input_width, image_height = image.shape[:2]
    kernel_width, kernel_height = kernel.shape
    output_shape = ((input_width - kernel_width) // stride + 1, (image_height - kernel_height) // stride + 1)
    strides = (stride * image.strides[0], stride * image.strides[1]) + image.strides[:2]
    strided_input = as_strided(image, shape=output_shape + (kernel_width, kernel_height), strides=strides)
    return np.max(strided_input, axis=(2, 3))


def laplace_fil():
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered_image = np.dstack(
        [convolve(image2[:, :, channel], laplace_filter, mode="constant", cval=0.0) for channel in range(3)])
    return filtered_image


def rozmywanie(image):
    filter = np.array([[1, 1, 1], [2, 4, 2], [1, 1, 1]]) * (1 / 16)
    filtered_image = convolve(image, filter, mode="constant", cval=0.0)
    return filtered_image


def wyostrzenie(image):
    sharp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filtered_image = convolve(image, sharp_filter, mode="constant", cval=0.0)
    return filtered_image


def sobel(image):
    S_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    S_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edges_x = convolve(image, S_x, mode='reflect')
    edges_y = convolve(image, S_y, mode='reflect')
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
    edges = edges / edges.max()
    return edges


pooled = max_pool(image_gray, kernel=np.zeros([2, 2]), stride=1)
pooled2 = max_pool(image_gray, kernel=np.zeros([7, 7]), stride=1)
filtered_image_laplace = laplace_fil()
filtr_sobel = sobel(image2_gray)
filtered_image_blur = rozmywanie(image_gray)
filtered_blured = rozmywanie(pooled2)
filtered_image_sharpen = wyostrzenie(pooled)
filtered_image_sharpen = np.clip(filtered_image_sharpen, 0, 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Oryginalny obraz')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image_sharpen, cmap='gray')
plt.title('Obraz po wyostrzeniu')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image_blur, cmap="gray")
plt.title('Obraz po rozmyciu')
plt.axis('off')

plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image2)
plt.title('Obraz oryginalny')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtr_sobel, cmap='gray')
plt.title('Obraz po filtrze Sobla')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image_laplace)
plt.title('Obraz po filtrze Laplace')
plt.axis('off')
plt.show()


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Obraz oryginalny')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image_blur, cmap="gray")
plt.title('Obraz po rozmyciu')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_blured, cmap='gray')
plt.title('Rozmycie Gausowskie z rozmiarem jądra 7 na 7')
plt.axis('off')
plt.show()