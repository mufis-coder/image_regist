from image_regist.tools import findTransformation, Transform
from PIL import Image
img1 = Image.open("resource\Elang_Jawa.jpg")
img2 = Image.open("resource\Elang_Jawa.jpg")


trans_x = 20
trans_y = 40
img2 = img2.rotate(45)
img2 = img2.transform(img2.size, Image.Transform.AFFINE,
                      (1, 0, trans_x, 0, 1, trans_y))

img3 = img2.transform(img2.size, Image.Transform.AFFINE, (1, 0, 19.68641438, 0, 1, -57.44388179))
img3 = img3.rotate(-47.36936165)

img1.save("test1.jpg")
img2.save("test2.jpg")
img3.save("test3.jpg")

import numpy as np

# findTransformation(img1, img2, [Transform.TRANSLATION_X, Transform.ROTATION, Transform.TRANSLATION_Y])