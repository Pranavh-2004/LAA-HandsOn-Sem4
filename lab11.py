import numpy as np  
from PIL import Image
import os
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Project 11: Projections, eigenvectors, Principal Component Analysis, and face recognition algorithms

print("\nSubtask 1\n")
# Parameters
Database_Size = 30
database_path = "/Users/pranavhemanth/Code/Academics/LA-S4/database"
# Initialize list to store image vectors
P = []
# Reading images from the database
for j in range(1, Database_Size + 1):
    image_path = os.path.join(database_path, f'person{j}.pgm')
    image = Image.open(image_path)
    image_array = np.array(image)
    # Get dimensions of the image
    m, n = image_array.shape
    # Reshape the image array to a column vector
    image_vector = image_array.reshape(m * n, 1)
    P.append(image_vector)
# Convert list to numpy array (matrix)
P = np.hstack(P)
# Print out the variables for verification
print(f"Database Size: {Database_Size}")
print(f"Image dimensions (m, n): ({m}, {n})")
print(f"P matrix shape: {P.shape}")

print("\nSubtask 2\n")
# Compute the mean face
mean_face = np.mean(P, axis=1)
# Reshape the mean face back to the original image dimensions
mean_face_image = mean_face.reshape(m, n)
# Display the mean face image
plt.imshow(mean_face_image, cmap='gray')
plt.title('Mean Face')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 3\n")
# Compute the mean face
mean_face = np.mean(P, axis=1)
# Convert P to double (float64 in numpy)
P = P.astype(np.float64)
# Subtract the mean face from each column of P
mean_face_column = mean_face.reshape(-1, 1)
P = P - mean_face_column @ np.ones((1, Database_Size))
# Print the first column of P to verify subtraction
print(P[:, 0])

print("\nSubtask 4\n")
# Compute the covariance matrix P^T * P
PTP = P.T @ P
# Compute the eigenvalues and eigenvectors of P^T * P
Values, Vectors = np.linalg.eig(PTP)
# Compute the actual eigenvectors of the covariance matrix
EigenVectors = P @ Vectors
# Normalize the eigenvectors
EigenVectors = EigenVectors / np.linalg.norm(EigenVectors, axis=0)
# Display the first few eigenvalues for verification
print("Eigenvalues:", Values)

print("\nSubtask 5\n")
# Display the set of eigenfaces
eigenfaces = []
for j in range(1, Database_Size):
    eigenface = EigenVectors[:, j] + mean_face
    eigenface_image = eigenface.reshape(m, n)
    eigenfaces.append(eigenface_image)
# Concatenate the eigenfaces horizontally
EigenFaces = np.hstack(eigenfaces)
# Display the eigenfaces
plt.figure(figsize=(15, 5))
plt.imshow(EigenFaces, cmap='gray')
plt.title('Eigenfaces')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 6\n")
# Compute the Products matrix
Products = EigenVectors.T @ EigenVectors
# Print the Products matrix to verify orthogonality
print("Products matrix:")
print(Products)
# Check if Products matrix is diagonal
is_diagonal = np.allclose(Products, np.diag(np.diagonal(Products)))
print(f"Is Products matrix diagonal? {is_diagonal}")

print("\nSubtask 7\n")
# Define image dimensions
m, n = 112, 92
# Read the altered image
altered_image_path ="/Users/pranavhemanth/Code/Academics/LA-S4/database/person30altered1.pgm"
image_read = Image.open(altered_image_path)
image_array = np.array(image_read)
U = image_array.reshape(m * n, 1)
# Compute the norms of the eigenvectors
Products = EigenVectors.T @ EigenVectors
NormsEigenVectors = np.diag(Products)
# Compute the projection coefficients
W = EigenVectors.T @ (U.astype(np.float64) - mean_face.reshape(-1, 1))
W = W / NormsEigenVectors.reshape(-1, 1) # Ensure proper division
# Reconstruct the image from the projection
U_approx = EigenVectors @ W + mean_face.reshape(-1, 1)
# Print shapes for debugging
print("Shape of U_approx:", U_approx.shape)
print("Expected shape:", (m * n, 1))
# Ensure the shape matches for reshaping
if U_approx.shape[0] == m * n and U_approx.shape[1] == 1:
    image_approx = U_approx.reshape(m, n).astype(np.uint8)
else:
    raise ValueError(f"Cannot reshape array of size {U_approx.size} into shape ({m}, {n})")
# Display the original altered image and the reconstructed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Altered Image')
plt.axis('off') # Hide axis
plt.subplot(1, 2, 2)
plt.imshow(image_approx, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 8\n")
# Define image dimensions
m, n = 112, 92
# Read the altered image
altered_image_path ="/Users/pranavhemanth/Code/Academics/LA-S4/database/person31.pgm"
image_read = Image.open(altered_image_path)
image_array = np.array(image_read)
U = image_array.reshape(m * n, 1)
# Compute the norms of the eigenvectors
Products = EigenVectors.T @ EigenVectors
NormsEigenVectors = np.diag(Products)
# Compute the projection coefficients
W = EigenVectors.T @ (U.astype(np.float64) - mean_face.reshape(-1, 1))
W = W / NormsEigenVectors.reshape(-1, 1) # Ensure proper division
# Reconstruct the image from the projection
U_approx = EigenVectors @ W + mean_face.reshape(-1, 1)
# Print shapes for debugging
print("Shape of U_approx:", U_approx.shape)
print("Expected shape:", (m * n, 1))
# Ensure the shape matches for reshaping
if U_approx.shape[0] == m * n and U_approx.shape[1] == 1:
    image_approx = U_approx.reshape(m, n).astype(np.uint8)
else:
    raise ValueError(f"Cannot reshape array of size {U_approx.size} into shape ({m}, {n})")
# Display the original altered image and the reconstructed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Altered Image')
plt.axis('off') # Hide axis
plt.subplot(1, 2, 2)
plt.imshow(image_approx, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off') # Hide axis
plt.show()

print("\nSubtask 9\n")
# # Recognition and approximation of a new face (person31.pgm)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Define image dimensions
m, n = 112, 92
# Read the new image (person31.pgm)
new_image_path ="/Users/pranavhemanth/Code/Academics/LA-S4/database/person31.pgm"
image_read = Image.open(new_image_path)
image_array = np.array(image_read)
U = image_array.reshape(m * n, 1)
# Compute the norms of the eigenvectors
Products = EigenVectors.T @ EigenVectors
NormsEigenVectors = np.diag(Products)
# Compute the projection coefficients
W = EigenVectors.T @ (U.astype(np.float64) - mean_face.reshape(-1, 1))
W = W / NormsEigenVectors.reshape(-1, 1) # Ensure proper division
# Reconstruct the image from the projection
U_approx = EigenVectors @ W + mean_face.reshape(-1, 1)
114
# Print shapes for debugging
print("Shape of U_approx:", U_approx.shape)
print("Expected shape:", (m * n, 1))
# Ensure the shape matches for reshaping
if U_approx.shape[0] == m * n and U_approx.shape[1] == 1:
    image_approx = U_approx.reshape(m, n).astype(np.uint8)
else:
    raise ValueError(f"Cannot reshape array of size {U_approx.size} into shape ({m}, {n})")
# Display the original new image and the reconstructed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original New Image')
plt.axis('off') # Hide axis
plt.subplot(1, 2, 2)
plt.imshow(image_approx, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off') # Hide axis
plt.show()
# Variables: image_read, U, NormsEigenVectors, W, U_approx