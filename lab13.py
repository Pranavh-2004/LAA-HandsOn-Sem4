import numpy as np  
import matplotlib.pyplot as plt
import scipy.io
from scipy.linalg import eigh
from scipy.io import loadmat

# Project 13: Social networks, clustering, and eigenvalue problems

print("\nSubtask 1\n")
# 1. Simple graph: Define adjacency matrix
AdjMatrix = np.array([[0, 1, 1, 0],
[1, 0, 0, 1],
[1, 0, 0, 1],
[0, 1, 1, 0]])
print("Adjacency Matrix:")
print(AdjMatrix)

print("\nSubtask 2\n")
# 2. Find the row sums of the matrix AdjMatrix
RowSums = np.sum(AdjMatrix, axis=1)
print("\nRow Sums:")
print(RowSums)

print("\nSubtask 3\n")
# 3. Compute the Laplacian of the graph
LaplaceGraph = np.diag(RowSums) - AdjMatrix
print("\nLaplacian Matrix:")
print(LaplaceGraph)
# Check if LaplaceGraph is singular
test_vector = np.ones(len(LaplaceGraph))
singularity_check = LaplaceGraph @ test_vector
print("\nSingularity Check (Laplacian * ones):")
print(singularity_check)

print("\nSubtask 4\n")
# 4. Find eigenvalues and eigenvectors using the eig function
D, V = np.linalg.eig(LaplaceGraph)

print("\nSubtask 5\n")
d, ind = np.argsort(D), np.argsort(D)
D = np.diag(D[ind])
V = V[:, ind]
print("\nEigenvalues (sorted):")
print(np.diag(D))
print("\nEigenvectors (sorted):")
print(V)

print("\nSubtask 6\n")
# 6. Identify the second smallest eigenvalue and its corresponding eigenvector
second_smallest_eigenvalue = D[1, 1]
V2 = V[:, 1]
# Ensure V2 has a positive first entry
if V2[0] < 0:
    V2 = -V2
print("\nSecond Smallest Eigenvalue:")
print(second_smallest_eigenvalue)
print("\nEigenvector corresponding to the second smallest eigenvalue (V2):")
print(V2)

print("\nSubtask 7\n")
# 7. Separate the elements of the eigenvector V2
pos = []
neg = []
for j in range(len(V2)):
    if V2[j] > 0:
        pos.append(j)
    else:
        neg.append(j)
print("\nPositive Indices (V2 > 0):")
print(pos)
print("\nNegative Indices (V2 <= 0):")
print(neg)
# Optional: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(pos, [1]*len(pos), color='green', label='Positive')
plt.scatter(neg, [0]*len(neg), color='red', label='Negative')
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.title('Clustering based on Eigenvector V2')
plt.xlabel('Indices')
plt.legend()
plt.grid()
plt.show()

print("\nSubtask 8\n")
# 8. Load the data
data = loadmat("/Users/pranavhemanth/Code/Academics/LA-S4/social.mat")
Social = data['Social']
print("Loaded Social adjacency matrix with shape:", Social.shape)
# Spy plot of the Social matrix
plt.figure(figsize=(8, 6))
plt.spy(Social, markersize=1)
plt.title('Sparsity pattern of the Social adjacency matrix')
plt.show()

print("\nSubtask 9\n")
# 9. Define DiagSocial and LaplaceSocial
DiagSocial = np.sum(Social, axis=1)
LaplaceSocial = np.diag(DiagSocial) - Social
print("\nDiagonal matrix DiagSocial:")
print(DiagSocial)
print("\nLaplacian matrix LaplaceSocial:")
print(LaplaceSocial)

print("\nSubtask 10\n")
# 10. Compute eigenvalues and eigenvectors
D, V = np.linalg.eig(LaplaceSocial)
print("\nEigenvalues (D):")
print(D)
print("\nEigenvectors (V):")
print(V)
# Check the shapes
print("Shape of V (eigenvectors):", V.shape)
print("Shape of D (eigenvalues):", D.shape)
d, ind = np.argsort(D), np.argsort(D)
D = np.diag(D[ind])
V = V[:, ind]
print("\nEigenvalues (sorted):")
print(np.diag(D))
print("\nEigenvectors (sorted):")
print(V)


print("\nSubtask 11\n")
second_smallest_eigenvalue = D[1, 1]
V2 = V[:, 1]
# Ensure V2 has a positive first entry
if V2[0] < 0:
    V2 = -V2
print("\nSecond Smallest Eigenvalue:")
print(second_smallest_eigenvalue)
print("\nEigenvector corresponding to the second smallest eigenvalue (V2):")
print(V2)
pos = []
neg = []
for j in range(len(V2)):
    if V2[j] > 0:
        pos.append(j)
    else:
        neg.append(j)
print("\nPositive Indices (V2 > 0):")
print(pos)
print("\nNegative Indices (V2 <= 0):")
print(neg)

print("\nSubtask 12\n")
# Create the order based on positive and negative indices
order = pos + neg # Combine the positive and negative indices
m, n = Social.shape # Get the shape of the Social matrix
iden = np.eye(m) # Identity matrix of size m
# Create the permutation matrix P
P = np.zeros((m, m))
for j in range(m):
    for k in range(m):
        P[j, k] = iden[order[j], k]
# Permute the adjacency matrix
SocialOrdered = P @ Social @ P.T # Using matrix multiplication
print("Shape of SocialOrdered:", SocialOrdered.shape)

print("\nSubtask 13\n")
# Plot the permuted adjacency matrix
plt.figure(figsize=(8, 6))
plt.spy(SocialOrdered, markersize=1) # Using a smaller marker size for better visibility
plt.title("Spy Plot of Permuted Adjacency Matrix (SocialOrdered)")
plt.xlabel("Nodes")
plt.ylabel("Nodes")
plt.grid(False) # Disable the grid
plt.show()

print("\nSubtask 14\n")
# Explore the third smallest eigenvalue for clustering
V3 = V[:, 2] # Get the third eigenvector
if V3[0] < 0: # Ensure V3 has a positive first entry
    V3 = -V3
# Initialize lists for the groups
pp = [] # ++ group
pn = [] # +- group
NP = [] # -+ group
nn = [] # -- group
# Grouping based on the signs of V2 and V3
for j in range(len(V2)):
    if V2[j] > 0:
        if V3[j] > 0:
            pp.append(j)
        else:
            pn.append(j)
    else:
        if V3[j] > 0:
            NP.append(j)
        else:
            nn.append(j)
# Combine the orders of the groups
order = pp + pn + NP + nn
m = len(Social) # Get the size of Social
iden = np.eye(m) # Identity matrix of size m
P = np.zeros((m, m)) # Initialize permutation matrix
# Create the permutation matrix
for j in range(m):
    P[j, :] = iden[order[j], :]
# Permute the adjacency matrix
SocialOrdered = P @ Social @ P.T
# Plot the permuted adjacency matrix
plt.figure(figsize=(8, 6))
plt.spy(SocialOrdered, markersize=1)
plt.title("Spy Plot of Permuted Adjacency Matrix (SocialOrdered)")
plt.xlabel("Nodes")
plt.ylabel("Nodes")
plt.grid(False) # Disable the grid
plt.show()

print("\nSubtask 15\n")
# Step 15: Fiedler vector procedure iteratively for clusters
import numpy as np
import matplotlib.pyplot as plt
# Assuming 'Social' is your adjacency matrix, and 'pos' and 'neg' are your positive and negative indices
# Define SocialPos and SocialNeg based on the positive and negative indices
SocialPos = Social[np.ix_(pos, pos)]
SocialNeg = Social[np.ix_(neg, neg)]
# Calculate the Laplacian for the positive group
rowsumpos = np.sum(SocialPos, axis=1)
DiagSocialPos = np.diag(rowsumpos)
LaplaceSocialPos = DiagSocialPos - SocialPos
# Eigen decomposition for positive group
DPos , VPos = np.linalg.eig(LaplaceSocialPos)
d, ind = np.argsort(DPos), np.argsort(DPos)
DPos = np.diag(DPos[ind])
VPos = VPos[:, ind]
V2Pos = VPos[:, 1] # Second smallest eigenvector for positive group
# Group positive nodes
posp = [] # Positive group
posn = [] # Negative group
for j in range(len(V2Pos)):
    if V2Pos[j] > 0:
        posp.append(pos[j]) # Append original index
    else:
        posn.append(pos[j]) # Append original index
# Calculate the Laplacian for the negative group
rowsumneg = np.sum(SocialNeg, axis=1)
DiagSocialNeg = np.diag(rowsumneg)
LaplaceSocialNeg = DiagSocialNeg - SocialNeg
# Eigen decomposition for negative group
DNeg , VNeg = np.linalg.eig(LaplaceSocialNeg)
d, ind = np.argsort(DNeg), np.argsort(DNeg)
DNeg = np.diag(DNeg[ind])
VNeg = VNeg[:, ind]
V2Neg = VNeg[:, 1] # Second smallest eigenvector for negative group
# Group negative nodes
negp = [] # Positive group
negn = [] # Negative group
for j in range(len(V2Neg)):
    if V2Neg[j] > 0:
        negp.append(neg[j]) # Append original index
    else:
        negn.append(neg[j]) # Append original index 
# Generate the final order for the permutation
ordergen = posp + posn + negp + negn
# Create the permutation matrix
m = len(Social) # Assuming the size of Social
iden = np.eye(m) # Identity matrix of size m
P = np.zeros((m, m)) # Initialize permutation matrix
# Create the permutation matrix
for j in range(m):
    P[j, :] = iden[ordergen[j], :] # Filling the permutation matrix based on ordergen
# Permute the adjacency matrix
SocialOrderedGen = P @ Social @ P.T # Permutation of the Social matrix
# Plot the permuted adjacency matrix
plt.figure(figsize=(10, 8))
plt.spy(SocialOrderedGen, markersize=1)
plt.title("Spy Plot of Permuted Adjacency Matrix (SocialOrderedGen)")
plt.xlabel("Nodes")
plt.ylabel("Nodes")
plt.grid(False) # Disable grid for clarity
plt.show()