import numpy as np  
import matplotlib.pyplot as plt
import scipy.io
import networkx as nx
from scipy.sparse import csr_matrix
from matplotlib.pylab import eig

# Project 12: Matrix eigenvalues and the Google’s PageRank algorithm

print("\nSubtask 1\n")
# Load the network data
# Load the adjacency matrix from the file 'AdjMatrix.mat'
data = scipy.io.loadmat("/Users/pranavhemanth/Code/Academics/LA-S4/AdjMatrix.mat")
AdjMatrix =csr_matrix(data['AdjMatrix'])
# Check the sparsity of the matrix
num_elements = AdjMatrix.shape[0] * AdjMatrix.shape[1]
num_non_zero_elements = AdjMatrix.nnz
nnzAdjMatrix = num_non_zero_elements / num_elements
print(f"Sparsity of AdjMatrix: {nnzAdjMatrix:.4f}")

print("\nSubtask 2\n")
# Check the dimensions of the matrix
m, n = AdjMatrix.shape
print(f"Dimensions of AdjMatrix: {m} x {n}")

print("\nSubtask 3\n")
# Create a smaller submatrix and plot the network
NumNetwork = 500
AdjMatrixSmall = AdjMatrix[:NumNetwork, :NumNetwork].toarray() # Extract submatrix
# Generate random coordinates for the nodes
#np.random.seed(0) # For reproducibility
coordinates = np.random.rand(NumNetwork, 2) * NumNetwork # Random coordinates
# Plot the graph
plt.figure(figsize=(10, 10))
plt.plot(coordinates[:, 0], coordinates[:, 1], 'k-*')
plt.title('Subgraph of the First 500 Nodes')
plt.xlabel('Random X Coordinate')
plt.ylabel('Random Y Coordinate')
plt.show()
# Variables
print(f"AdjMatrixSmall shape: {AdjMatrixSmall.shape}")
print(f"Coordinates shape: {coordinates.shape}")
print(f"NumNetwork: {NumNetwork}")

print("\nSubtask 4\n")
# Compute the Google Matrix
alpha = 0.15
GoogleMatrix = np.zeros((NumNetwork, NumNetwork))
# Check the amount of links originating from each webpage
NumLinks = np.sum(AdjMatrixSmall, axis=1)
for i in range(NumNetwork):
    if NumLinks[i] != 0:
        GoogleMatrix[i, :] = AdjMatrixSmall[i, :] / NumLinks[i]
    else:
        GoogleMatrix[i, :] = 1.0 / NumNetwork
GoogleMatrix = (1 - alpha) * GoogleMatrix + alpha * np.ones((NumNetwork,
NumNetwork)) / NumNetwork
# Compute the vectors w0, w1, w2, w3, w5, w10
w0 = np.ones(NumNetwork) / np.sqrt(NumNetwork)
w1 = w0 @ GoogleMatrix
w2 = w1 @ GoogleMatrix
w3 = w2 @ GoogleMatrix
w10 = w0 @ (GoogleMatrix ** 10)
w5 = w0 @ (GoogleMatrix ** 5)
deltaw = w10 - w5
print("Difference δw:", np.linalg.norm(deltaw))

print("\nSubtask 5\n")
# Compute eigenvalues and eigenvectors
eigenvalues, right_eigenvectors = eig(GoogleMatrix)
# Find the index of the eigenvalue λ1 = 1
lambda_1_index = np.isclose(eigenvalues, 1)
# Get the right eigenvector corresponding to λ1
v1 = right_eigenvectors[:, lambda_1_index].flatten()
# Compute the left eigenvectors
left_eigenvalues, left_eigenvectors = eig(GoogleMatrix.T)
# Get the left eigenvector corresponding to λ1
u1 = left_eigenvectors[:, lambda_1_index].flatten()
print("Left Eigenvector (u1):", u1)

print("\nSubtask 6\n")
# Normalize u1 to have all positive components
u1 = np.abs(u1) / np.linalg.norm(u1, 1)

print("\nSubtask 7\n")
MaxRank, PageMaxRank = np.max(u1), np.argmax(u1)
print(f"MaxRank: {MaxRank}, PageMaxRank: {PageMaxRank}")

print("\nSubtask 8\n")
MostLinks = np.sum(AdjMatrixSmall, axis=0) # Sum of columns
MaxLinks, PageMaxLinks = np.max(MostLinks), np.argmax(MostLinks)
print(f"MostLinks: {MostLinks}, MaxLinks: {MaxLinks}, PageMaxLinks: {PageMaxLinks}")

print("\nSubtask 9\n")
are_equal = PageMaxRank == PageMaxLinks
print(f"Is the highest ranking webpage the same as the page with the most hyperlinks?{are_equal}")
# Q1: What is the number of hyperlinks pointing to the webpage MaxRank?
print(f"Number of hyperlinks pointing to the webpage MaxRank:{MostLinks[PageMaxRank]}")
