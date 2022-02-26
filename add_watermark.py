import cv2
from matplotlib import pyplot as plt
from watermark import add_watermark

img = cv2.imread("./pictures/avatar.jpg", 1)
watermark = cv2.imread("./pictures/watermark.jpg", 0)

img_watermark = add_watermark(img, watermark, 20000, 8)

# print image
plt.subplot(1, 2, 1), plt.imshow(img[:, :, [2, 1, 0]])
plt.title("input image"), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(img_watermark[:, :, [2, 1, 0]])
plt.title("watermark"), plt.xticks([]), plt.yticks([])

plt.show()

cv2.imwrite("./pictures/output.jpg", img_watermark)
