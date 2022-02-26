import cv2
import numpy as np


def decrypt_single(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    spectrum = np.log(np.abs(f_shift))
    maxx = np.max(spectrum)
    minn = np.min(spectrum)
    spectrum = (spectrum - minn) / (maxx - minn) * 255
    return spectrum


def decrypt(img):
    img_b, img_g, img_r = cv2.split(img)
    spectrum_b = decrypt_single(img_b)
    spectrum_g = decrypt_single(img_g)
    spectrum_r = decrypt_single(img_r)
    return spectrum_b, spectrum_g, spectrum_r


def add_watermark_single(img, watermark, coefficient):
    h, w = watermark.shape

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_shift[0:h, 0:w] += watermark * coefficient
    f_shift[-h:, -w:] += cv2.flip(watermark, -1) * coefficient

    if_shift = np.fft.ifftshift(f_shift)
    img_watermark = np.fft.ifft2(if_shift)
    img_watermark = np.abs(img_watermark)
    img_watermark = np.uint8(img_watermark)
    return img_watermark


def add_watermark(img, watermark, coefficient, ratio=8):
    img_b, img_g, img_r = cv2.split(img)
    img_h, img_w = img.shape[0], img.shape[1]
    wm_h, wm_w = watermark.shape[0], watermark.shape[1]
    ratio_h = img_h / wm_h / ratio
    ratio_w = img_w / wm_w / ratio
    ratio = ratio_h if ratio_w > ratio_h else ratio_w
    wm_h = np.int(wm_h * ratio)
    wm_w = np.int(wm_w * ratio)

    watermark = cv2.resize(watermark, (wm_w, wm_h))
    watermark = 255 - watermark
    img_b_watermark = add_watermark_single(img_b, watermark, coefficient)
    img_g_watermark = add_watermark_single(img_g, watermark, coefficient)
    img_r_watermark = add_watermark_single(img_r, watermark, coefficient)

    img_watermark = cv2.merge([img_b_watermark, img_g_watermark, img_r_watermark])
    return img_watermark
