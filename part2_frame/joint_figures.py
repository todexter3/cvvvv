import os
import numpy as np
from PIL import Image
# Set the path to the directory containing the images
path_to_images = "filters/"

# Get a list of all the image filenames in the directory
image_filenames = os.listdir(path_to_images)

# Sort the filenames so that the images are in the correct order
image_filenames.sort()

# Create an empty array to hold the final image
final_image = np.zeros((6*64, 6*64, 3), dtype=np.uint8)

# Loop over the images and add them to the final image
for i, filename in enumerate(image_filenames):
    # Load the image
    image = np.array(Image.open(os.path.join(path_to_images, filename)))

    # Calculate the row and column indices for this image
    row = i // 6
    col = i % 6

    # Add the image to the final image
    final_image[row*64:(row+1)*64, col*64:(col+1)*64, :] = image

# Save the final image
Image.fromarray(final_image).save("final_image.png")

