import numpy as np  
import matplotlib.pyplot as plt

# Project 14: Singular Value Decomposition and image compression

print("\nSubtask 1\n")
# Task 1: Plotting the unit circle and basis vectors
t = np.linspace(0, 2 * np.pi, 100)
X = np.array([np.cos(t), np.sin(t)])
plt.subplot(2, 2, 1)
plt.plot(X[0, :], X[1, :], 'b')
plt.quiver(0, 0, 1, 0, color='r', angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, 0, 1, color='g', angles='xy', scale_units='xy', scale=1)
plt.axis('equal')
plt.title('Unit circle')
plt.show()

print("\nSubtask 2\n")
A = np.array([[2, 1], [-1, 1]])
U, S, V = np.linalg.svd(A)
print("U:\n", U)
print("S:\n", S)
print("V:\n", V)
# Verify orthogonality
print("U' * U:\n", np.dot(U.T, U))
print("V' * V:\n", np.dot(V.T, V))

print("\nSubtask 3\n")
VX = np.dot(V.T, X)
plt.subplot(2, 2, 2)
plt.plot(VX[0, :], VX[1, :], 'b')
plt.quiver(0, 0, V[0, 0], V[0, 1], color='r', angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, V[1, 0], V[1, 1], color='g', angles='xy', scale_units='xy', scale=1)
plt.axis('equal')
plt.title('Multiplied by matrix V^T')
plt.show()

print("\nSubtask 4\n")
S_matrix = np.diag(S)
SVX = np.dot(S_matrix, VX)
plt.subplot(2, 2, 3)
plt.plot(SVX[0, :], SVX[1, :], 'b')
plt.quiver(0, 0, S[0] * V[0, 0], S[1] * V[0, 1], color='r', angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, S[0] * V[1, 0], S[1] * V[1, 1], color='g', angles='xy', scale_units='xy', scale=1)
plt.axis('equal')
plt.title('Multiplied by matrix ΣV^T')
plt.show()

print("\nSubtask 5\n")
AX = np.dot(U, SVX)
plt.subplot(2, 2, 4)
plt.plot(AX[0, :], AX[1, :], 'b')
plt.quiver(0, 0, U[0, 0] * S[0] * V[0, 0] + U[0, 1] * S[1] * V[0, 1],
U[1, 0] * S[0] * V[0, 0] + U[1, 1] * S[1] * V[0, 1], color='r', angles='xy',
scale_units='xy', scale=1)
plt.quiver(0, 0, U[0, 0] * S[0] * V[1, 0] + U[0, 1] * S[1] * V[1, 1],
U[1, 0] * S[0] * V[1, 0] + U[1, 1] * S[1] * V[1, 1], color='g', angles='xy',
scale_units='xy', scale=1)
plt.axis('equal')
plt.title('Multiplied by matrix UΣV^T=A')
plt.show()

print("\nSubtask 6\n")
# Modification example for U and V (this is just a random example, modifications need to be chosen carefully)
U1 = U
V1 = V.T
print("U1 * S * V1.T:\n", np.dot(U1, np.dot(S_matrix, V1.T)))

print("\nSubtask 7\n")
Av1 = np.dot(A, V.T[:, 0])
Av2 = np.dot(A, V.T[:, 1])
print("Av1:\n", Av1)
print("σ1 * u1:\n", S[0] * U[:, 0])
print("Av2:\n", Av2)
print("σ2 * u2:\n", S[1] * U[:, 1])
# Numerical check
print("A * V - U * S:\n", np.dot(A, V.T) - np.dot(U, S_matrix))

print("\nSubtask 8-11\n")
import cv2
# Load the image
ImJPG = cv2.imread("/Users/pranavhemanth/Code/Academics/LA-S4/Albert_Einstein_Head.jpg",cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Image')
plt.show()
# Singular Value Decomposition
UIm, SIm, VIm = np.linalg.svd(ImJPG.astype(np.float64), full_matrices=False)
print(UIm)
# Plot Singular Values
plt.figure()
plt.plot(np.arange(len(SIm)), SIm)
plt.title('Singular Values')
plt.show()
# Image compression using truncated SVD
for k in [50, 100, 150]:
    ImJPG_comp = np.dot(UIm[:, :k], np.dot(np.diag(SIm[:k]), VIm[:k, :]))
    plt.figure()
    plt.imshow(ImJPG_comp, cmap='gray')
    plt.title(f'Compressed Image with {k} Singular Values')
    plt.show()
    pct = 1 - (np.size(UIm[:, :k]) + np.size(VIm[:k, :]) * np.size(np.diag(SIm[:k]))) /np.size(ImJPG)
    print(f'Compression percentage for {k} singular values: {pct:.3f}')

print("\nSubtask 12\n")
#12
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Load the image
ImJPG = cv2.imread("/Users/pranavhemanth/Code/Academics/LA-S4/checkers.pgm",cv2.IMREAD_GRAYSCALE)
# Add noise to the image
m, n = ImJPG.shape
ImJPG_Noisy = ImJPG.astype(np.float64) + 50 * (np.random.rand(m, n) - 0.5)
ImJPG_Noisy = np.clip(ImJPG_Noisy, 0, 255) # Ensure values are within valid range
# Display the original and noisy images
plt.figure()
plt.imshow(ImJPG, cmap='gray')
plt.title('Original Checkers Image')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(ImJPG_Noisy, cmap='gray')
plt.title('Noisy Checkers Image')
plt.axis('off')
plt.show()

print("\nSubtask 13\n")
# Compute SVD of the noisy image
UIm, SIm, VIm = np.linalg.svd(ImJPG_Noisy, full_matrices=False)
# Variables: UIm, SIm, VIm

print("\nSubtask 14\n")
# Function to approximate the image with k singular values
def approximate_image(U, S, V, k):
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
# Approximations with k = 10, k = 30, k = 50 singular values
ks = [10, 30, 50]
for k in ks:
    ImJPG_approx = approximate_image(UIm, SIm, VIm, k)
    plt.figure()
    plt.imshow(ImJPG_approx, cmap='gray')
    plt.title(f'Denoised Image with k = {k} Singular Values')
    plt.axis('off')
    plt.show()
    # Compare the images to the initial noisy image
    plt.figure()
    plt.imshow(np.hstack((ImJPG, ImJPG_Noisy, ImJPG_approx)), cmap='gray')
    plt.title(f'Original, Noisy, and Denoised (k = {k}) Images')
    plt.axis('off')
    plt.show()