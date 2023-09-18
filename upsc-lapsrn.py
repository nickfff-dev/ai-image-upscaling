from PIL import Image
import cv2
import os


sr = cv2.dnn_superres.DnnSuperResImpl_create()
imgName = input('Enter the image name plus extension(like image.png): ')
image = cv2.imread(imgName)
# Read image


# Read the desired model

path = "LapSRN_x8.pb"

sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing

sr.setModel("lapsrn", 8)

# Upscale the image
result = sr.upsample(image)

# Save the image

cv2.imwrite(f"{imgName}-lapsrn.png", result)
