from PIL import Image
import numpy as np
img1 = Image.open("resource\Elang_Jawa.jpg")
img2 = Image.open("resource\Elang_Jawa.jpg")

from image_regist.tools import mutual_information_2d

print(len(np.asarray(img1).ravel()))
print(mutual_information_2d(np.asarray(img1).ravel(), np.asarray(img2).ravel()))