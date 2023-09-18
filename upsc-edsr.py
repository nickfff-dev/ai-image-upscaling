from PIL import Image
import cv2
import os


sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Read image
imgName = input('Enter the image name plus extension(like image.png): ')
image = cv2.imread(imgName)

# Read the desired model

path = "EDSR_x4.pb"

sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing

sr.setModel("edsr", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image

cv2.imwrite(f"{imgName}-edsr.png", result)
