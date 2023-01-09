'''Compare two images and print the difference in a matrix format.'''


from PIL import Image
import numpy as np
import os
img1 = np.array(Image.open("color_lytro_06_new.png"))
img2 = np.array(Image.open("data/result/color_lytro_06.png"))
np.set_printoptions(threshold=np.inf, linewidth=520, precision=0)
diff = img1[:,:,:3] - img2
diff = diff.sum(axis=2)

print(diff)