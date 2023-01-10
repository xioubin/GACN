'''Compare two images and print the difference in a matrix format.'''


from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

img1 = cv2.imread("color_lytro_20_new.png")
img2 = cv2.imread("data/result/color_lytro_20.png")

# PSNR

# MSE = cv2.mse(img1, img2)
PSNR_self = cv2.PSNR(img2, img2)
PSNR = cv2.PSNR(img1, img2)

# SSIM

SSIM = ssim(img1, img2, multichannel=True)

# print("MSE: ", MSE)
print("PSMR_self: ", PSNR_self)
print("PSNR: ", PSNR)
print("SSIM: ", SSIM)