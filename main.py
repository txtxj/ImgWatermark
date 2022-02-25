import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./avatar.jpg", 0)
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
plt.title("input image"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("magnitude spectrum"), plt.xticks([]), plt.yticks([])

plt.show()
