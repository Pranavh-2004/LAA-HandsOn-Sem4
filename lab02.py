import imageio.v3 as iio  # imageio.v3 is the latest version for imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Project 2: Matrix operations and image manipulation

print("\nSubtask 1\n")
# Load a grayscale jpg file and represent the data as a matrix
ImJPG=iio.imread("Albert_Einstein_Head.jpg");
'''
plt.imshow(ImJPG, cmap="gray")  # Ensure grayscale display
plt.show()
'''

print("\nSubtask 2\n")
# Get the dimensions of the image
m, n = ImJPG.shape
# Print the dimensions
print(f'The dimensions of the image are {m} x {n}')
# Optional: Visualize the image
plt.imshow(ImJPG, cmap='gray')
plt.title('Einstein Grayscale Image')
plt.axis('off')
plt.show()

print("\nSubtask 3\n")
# Check if the array is of integer type
isInt = np.issubdtype(ImJPG.dtype, np.integer)
# Print the result
print(f'Is ImJPG of integer type? {isInt}')
# Optional: Visualize the image
'''
plt.imshow(ImJPG, cmap='gray')
plt.title('Einstein Grayscale Image')
plt.axis('off')
plt.show()
'''

print("\nSubtask 4\n")
isInt = np.issubdtype(ImJPG.dtype, np.integer)
# Find the range of colors in the image
maxImJPG = np.max(ImJPG)
minImJPG = np.min(ImJPG)
# Print the results
print(f'The dimensions of the image are {m} x {n}')
print(f'Is ImJPG of integer type? {isInt}')
print(f'The maximum pixel value in the image is {maxImJPG}')
print(f'The minimum pixel value in the image is {minImJPG}')
# Optional: Visualize the image
'''
plt.imshow(ImJPG, cmap='gray')
plt.title('Einstein Grayscale Image')
plt.axis('off')
plt.show()
'''

print("\nSubtask 5\n")
# Check if the array is of integer type
isInt = np.issubdtype(ImJPG.dtype, np.integer)
# Find the range of colors in the image
maxImJPG = np.max(ImJPG)
minImJPG = np.min(ImJPG)
# Print the results
print(f'The dimensions of the image are {m} x {n}')
print(f'Is ImJPG of integer type? {isInt}')
print(f'The maximum pixel value in the image is {maxImJPG}')
print(f'The minimum pixel value in the image is {minImJPG}')
# Display the image
plt.imshow(ImJPG, cmap='gray')
plt.title('Einstein Grayscale Image')
plt.axis('off')
plt.show()

print("\nSubtask 6\n")
# Get the dimensions of the image
m, n = ImJPG.shape
# Crop the central part of the image
ImJPG_center = ImJPG[100:m-100, 100:n-70]
# Display the cropped image
plt.figure()
plt.imshow(ImJPG_center, cmap='gray')
plt.title('Cropped Central Part of Einstein Image')
plt.axis('off')
plt.show()

print("\nSubtask 7\n")
# Get the dimensions of the image
m, n = ImJPG.shape
# Crop the central part of the image
ImJPG_center = ImJPG[100:m-100, 100:n-70]
# Create a zero matrix of type uint8 with the same dimensions as the original image
ImJPG_border = np.zeros((m, n), dtype=np.uint8)
# Paste the cropped image into the zero matrix
ImJPG_border[100:m-100, 100:n-70] = ImJPG_center
# Display the resulting image
plt.figure()
plt.imshow(ImJPG_border, cmap='gray')
plt.title('Image with Pasted Center')
plt.axis('off')
plt.show()

print("\nSubtask 8\n")
# Perform vertical flipping by reversing the rows of the matrix
ImJPG_vertflip = np.flipud(ImJPG)
# Display the original and flipped images side by side for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG_vertflip, cmap='gray')
plt.title('Vertically Flipped Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 9\n")
ImJPG = np.array(ImJPG)
# Transpose the image matrix
ImJPG_transpose = ImJPG.T
# Display the original and transposed images side by side for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG_transpose, cmap='gray')
plt.title('Transposed Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 10\n")
# Transpose the image matrix
ImJPG_transpose = ImJPG.T
# Flip the transposed image horizontally (along the vertical axis)
ImJPG_horflip = np.fliplr(ImJPG_transpose)
# Transpose the flipped image back to its original orientation
ImJPG_horflip = ImJPG_horflip.T
# Display the original and horizontally flipped images side by side for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG_horflip, cmap='gray')
plt.title('Horizontally Flipped Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 11\n")
ImJPG = np.array(ImJPG)
# Rotate the image matrix by 90 degrees counterclockwise
ImJPG90 = np.rot90(ImJPG)
# Display the original and rotated images side by side for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG90, cmap='gray')
plt.title('90 Degrees Rotated Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 12\n")
# Perform color inversion
ImJPG_inv = 255 - ImJPG
# Display the original and inverted images side by side for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG_inv, cmap='gray')
plt.title('Inverted Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 13\n")
# Darken the image by subtracting a constant value
ImJPG_dark = ImJPG - 50
ImJPG_dark[ImJPG_dark < 0] = 0 # Ensure no negative values
# Lighten the image by adding a constant value
ImJPG_light = ImJPG + 50
ImJPG_light[ImJPG_light > 255] = 255 # Ensure values do not exceed 255
# Display the original, darkened, and lightened images side by side for comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(ImJPG_dark, cmap='gray')
plt.title('Darkened Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(ImJPG_light, cmap='gray')
plt.title('Lightened Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 14\n")
# Darken the image by subtracting 50
ImJPG_dark = ImJPG - 50
ImJPG_dark[ImJPG_dark < 0] = 0 # Ensure no negative values
# Lighten the image by adding 100
ImJPG_light_100 = ImJPG + 100
ImJPG_light_100[ImJPG_light_100 > 255] = 255 # Ensure values do not exceed 255
# Lighten the image by adding 50
ImJPG_light_50 = ImJPG + 50
ImJPG_light_50[ImJPG_light_50 > 255] = 255 # Ensure values do not exceed 255
# Arrange the images in a 2x2 matrix
top_row = np.concatenate((ImJPG, ImJPG_dark), axis=1)
bottom_row = np.concatenate((ImJPG_light_100, ImJPG_light_50), axis=1)
ImJPG_Warhol = np.concatenate((top_row, bottom_row), axis=0)
# Display the resulting block matrix as a single image
plt.figure(figsize=(10, 10))
plt.imshow(ImJPG_Warhol, cmap='gray')
plt.title('Andy Warhol Style Image')
plt.axis('off')
plt.show()

print("\nSubtask 15\n")
image_path = "/Users/pranavhemanth/Code/Academics/LA-S4/Albert_Einstein_Head.jpg"
im = Image.open(image_path).convert('L') # Convert to grayscale
ImJPG = np.array(im)
# Naive conversion to black and white
ImJPG_bw = np.uint8(255 * np.floor(ImJPG / 128))
# Display the original and black and white images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ImJPG_bw, cmap='gray')
plt.title('Black and White Image')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nSubtask 16\n")
# Reduce the number of shades from 256 to 8
# Step 1: Normalize the pixel values to the range [0, 1]
imjpg_normalized = ImJPG / 255.0
# Step 2: Scale the pixel values to the range [0, 7] and round them
imjpg_reduced = np.round(imjpg_normalized * 7)
# Step 3: Scale back to the range [0, 255] and convert to uint8
imjpg8_array = np.uint8(imjpg_reduced * (255 / 7))
# Convert the numpy array back to an image
imjpg8 = Image.fromarray(imjpg8_array)
# Display the image in a separate window
plt.imshow(imjpg8, cmap='gray')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 17\n")
# Increase the contrast by multiplying with a constant (e.g., 1.25)
contrast_factor = 1.25
imjpg_high_contrast_array = np.clip(ImJPG * contrast_factor, 0, 255).astype(np.uint8)
# Convert the numpy array back to an image
imjpg_high_contrast = Image.fromarray(imjpg_high_contrast_array)
# Display the high contrast image
plt.imshow(imjpg_high_contrast, cmap='gray')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 18\n")
# Perform gamma correction with gamma = 0.95
gamma_05 = 0.95
imjpg_gamma_05_array = np.clip((ImJPG / 255.0) ** gamma_05 * 255, 0,
255).astype(np.uint8)
# Perform gamma correction with gamma = 1.05
gamma_15 = 1.05
imjpg_gamma_15_array = np.clip((ImJPG / 255.0) ** gamma_15 * 255, 0,
255).astype(np.uint8)
# Convert the numpy arrays back to images
imjpg_gamma_05 = Image.fromarray(imjpg_gamma_05_array)
imjpg_gamma_15 = Image.fromarray(imjpg_gamma_15_array)
# Display the gamma-corrected images
plt.figure()
plt.imshow(imjpg_gamma_05, cmap='gray')
plt.title('Gamma Correction with Y = 0.95')
plt.axis('off') # Hide axis
plt.show()
plt.figure()
plt.imshow(imjpg_gamma_15, cmap='gray')
plt.title('Gamma Correction with Y = 1.05')
plt.axis('off') # Hide axis
plt.show()
