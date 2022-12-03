from PIL import Image
from image_regist.tools import Transform, findTransformation, transform_image_2d

reference_image = Image.open("resource\Elang_Jawa.jpg")
target_image = Image.open("resource\Elang_Jawa.jpg")

target_image = transform_image_2d(target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, Transform.TRANSLATION_Y], [40, -20, 3])

best_params = findTransformation(reference_image, target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, 
                                Transform.TRANSLATION_Y], True)

tranformed_image = transform_image_2d(target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, Transform.TRANSLATION_Y], 
                                best_params)

reference_image.save("01_reference_image.png")
target_image.save("02_target_image.png")
tranformed_image.save("03_tranformed_image.png")