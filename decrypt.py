import cv2
from matplotlib import pyplot as plt
from watermark import decrypt

img = cv2.imread("./pictures/attack.jpg", 1)

spectrum_b, spectrum_g, spectrum_r = decrypt(img)

# print image
plt.subplot(2, 3, 2), plt.imshow(img[:, :, [2, 1, 0]])
plt.title("input image"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4), plt.imshow(spectrum_b, cmap="Greys")
plt.title("spectrum_b"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5), plt.imshow(spectrum_g, cmap="Greys")
plt.title("spectrum_g"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 6), plt.imshow(spectrum_r, cmap="Greys")
plt.title("spectrum_r"), plt.xticks([]), plt.yticks([])

plt.show()

cv2.imwrite("./pictures/spectrum_b.jpg", spectrum_b)
cv2.imwrite("./pictures/spectrum_g.jpg", spectrum_g)
cv2.imwrite("./pictures/spectrum_r.jpg", spectrum_r)
