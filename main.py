from PIL import Image
from image_regist.tools import findTransformation, Transform, transform_image_2d

reference_image = Image.open("resource\Elang_Jawa.jpg")
target_image = Image.open("resource\Elang_Jawa.jpg")

target_image = transform_image_2d(target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, Transform.TRANSLATION_Y], [-45, -20, 3])

# best_params = findTransformation(reference_image, target_image, [Transform.ROTATION, 
#                                 Transform.TRANSLATION_X, 
#                                 Transform.TRANSLATION_Y])

tranformed_image = transform_image_2d(target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, Transform.TRANSLATION_Y], 
                                [ 44.98814041, 12.13631626, -15.76185011])

reference_image.save("01_reference_image.jpg")
target_image.save("02_target_image.jpg")
tranformed_image.save("03_tranformed_image.jpg")