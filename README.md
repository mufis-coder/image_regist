# Image Registration [![src-code](https://badgen.net/badge/github/code/blue?icon=github)](https://github.com/mufis-coder/image_regist)

The image-regist library is used to find the best parameters (rotation and translation of the x and y axes) to transform the target image to align with the reference image.

## Concept

![alt text](https://github.com/mufis-coder/image_regist/blob/main/resource/image-explanation.jpg)

Using the Particle Swarm Optimization (PSO) algorithm for optimization and Mutual Information as a measurement metric for two images. The image-regist library tries to find the best parameters (rotation and translation of the x and y axes) of the target image so that they can be aligned with the reference image.

## Installation

For installation, you can use pip

```bash
pip install image-regist
```

or clone from the repository

```bash
git clone https://github.com/mufis-coder/image_regist
```

## Requirements

Install the packages below according to the version listed to use the image-regist library

- numpy==1.21.6
- Pillow==9.3.0
- scipy==1.7.3

## Usage

```py
from PIL import Image
from image_regist.tools import Transform, findTransformation, transform_image_2d

# Load reference and target image
reference_image = Image.open("your-reference-image-file")
target_image = Image.open("your-target-image-file")

"""
Find best parameters to transform target image
----
In this example three transformations are used [Transform.ROTATION, Transform.TRANSLATION X, 
Transform.TRANSLATION_Y]. You can use less than three and you don't have to use them sequentially. 
The result of ---best_params--- is a list in the order according to the parameter ---params---.
---
If you want the algorithm to run faster, you can set parameter ---faster=True---
"""
best_params = findTransformation(data1=reference_image, data2=target_image, params=[Transform.ROTATION, 
                                Transform.TRANSLATION_X, 
                                Transform.TRANSLATION_Y], faster=False)

# Transform the target image according to the parameters that have been searched for
tranformed_image = transform_image_2d(target_image, [Transform.ROTATION, 
                                Transform.TRANSLATION_X, Transform.TRANSLATION_Y], 
                                best_params)

# Display transformed image
tranformed_image.show()
```
