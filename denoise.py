import numpy as np
from skimage import color, io
from matplotlib import pyplot as plt
import pywt


def filter_image_fft(img_fft, keep_fraction):
    r, c = img_fft.shape
    img_fft[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    img_fft[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    return np.real(np.fft.ifft2(img_fft))


def filter_image_color(img, keep_fraction=0.1):
    denoised_c_img = np.dstack([
        filter_image_fft(np.fft.fft2(img[:, :, channel]), keep_fraction)
        for channel in range(3)
    ])
    return denoised_c_img


def filter_image_gray(img, keep_fraction=0.1):
    img_gray = color.rgb2gray(img)
    img_fft = np.fft.fft2(img_gray)
    filtered_img_fft = filter_image_fft(img_fft, keep_fraction)
    return np.clip(filtered_img_fft, 0, 1)


def denoise_wavelet(image, wavelet, noise_sigma):
    levels = int(np.floor(np.log2(image.shape[0])))
    wc = pywt.wavedec2(image, wavelet, level=levels)
    arr, coeff_slices = pywt.coeffs_to_array(wc)
    arr = pywt.threshold(arr, noise_sigma, mode='soft')
    nwc = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(nwc, wavelet)


im = io.imread("Image_noise-example.jpg")
denoised_c_img = filter_image_color(im)
denoised_gray_img = filter_image_gray(im)

wavelet = pywt.Wavelet("haar")
gray_image = color.rgb2gray(im)
img_denoised_wavelet = denoise_wavelet(gray_image, wavelet, 0.0001)

plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(im)
plt.title('Oryginalny obraz')

plt.subplot(1, 4, 2)
plt.imshow(np.clip(denoised_c_img.astype(int), 0, 255))
plt.title('Odszumiony obraz kolorowy (Fourier)')

plt.subplot(1, 4, 3)
plt.imshow(denoised_gray_img, cmap='gray')
plt.title('Odszumiony obraz w skali szarości (Fourier)')

plt.subplot(1, 4, 4)
plt.imshow(img_denoised_wavelet, cmap='gray')
plt.title('Odszumiony obraz w skali szarości (Wavelet)')

plt.show()
