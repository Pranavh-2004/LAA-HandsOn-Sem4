import numpy as np

# Project 1: Basic operations with matrices in Python

print("\nSubtask 1\n")
A = np.array([[1, 2, -10, 4], [3, 4, 5, -6], [3, 3, -2, 5]])
B = np.array([3, 3, 4, 2])
print("A:\n", A)
print("B:\n", B)

#function to determine length
def length(matrix):
    return max(matrix.shape)

print("\nSubtask 2\n")
# Length of A and B
lengthA = length(A)
lengthB = length(B)
print("lengthA:", lengthA)
print("lengthB:", lengthB)

print("\nSubtask 3\n")
# Add B as the fourth row of A and create the new matrix C
C = np.vstack((A, B))
print("C:\n", C)

print("\nSubtask 4\n")
# Create D from rows 2, 3, 4 and columns 3, 4 of C
D = C[1:4, 2:4]
print("D:\n", D)

print("\nSubtask 5\n")
# Transpose D to create E
E = D.T
print("E:\n", E)

print("\nSubtask 6\n")
# Check the size of E
m, n = E.shape
print("m:", m)
print("n:", n)

print("\nSubtask 7\n")
# Create equally spaced vectors using arange and linspace
EqualSpaced = np.arange(0, 2 * np.pi, np.pi / 10)
EqualSpaced1 = np.linspace(0, 2 * np.pi, 21)
print("EqualSpaced:\n", EqualSpaced)
print("EqualSpaced1:\n", EqualSpaced1)

print("\nSubtask 8\n")
# Find the maximum and minimum in each column of A
maxcolA = np.max(A, axis=0)
mincolA = np.min(A, axis=0)
print("maxcolA:", maxcolA)
print("mincolA:", mincolA)

print("\nSubtask 9\n")
# Find the maximum and minimum in each row of A
maxrowA = np.max(A, axis=1)
minrowA = np.min(A, axis=1)
# Find the maximum and minimum elements in the entire matrix A
maxA = np.max(A)
minA = np.min(A)
print("maxrowA:", maxrowA)
print("minrowA:", minrowA)
print("maxA:", maxA)
print("minA:", minA)

print("\nSubtask 10\n")
# Calculate mean and sum in each column and row of A
meancolA = np.mean(A, axis=0)
meanrowA = np.mean(A, axis=1)
sumcolA = np.sum(A, axis=0)
sumrowA = np.sum(A, axis=1)
# Calculate mean and sum of all elements in A
meanA = np.mean(A)
sumA = np.sum(A)
print("meancolA:", meancolA)
print("meanrowA:", meanrowA)
print("sumcolA:", sumcolA)
print("sumrowA:", sumrowA)
print("meanA:", meanA)
print("sumA:", sumA)

print("\nSubtask 11\n")
# Create matrices F and G with random integers from -4 to 4
F = np.random.randint(-4, 5, (5, 3))
G = np.random.randint(-4, 5, (5, 3))
print("F:\n", F)
print("G:\n", G)

print("\nSubtask 12\n")
# Perform scalar multiplication, addition, subtraction, and element-wise multiplication on F and G
ScMultF = 0.4 * F
SumFG = F + G
DiffFG = F - G
ElProdFG = F * G
print("ScMultF:\n", ScMultF)
print("SumFG:\n", SumFG)
print("DiffFG:\n", DiffFG)
print("ElProdFG:\n", ElProdFG)

print("\nSubtask 13\n")
# Check the size of F and A
sizeF = F.shape
sizeA = A.shape
print("sizeF:", sizeF)
print("sizeA:", sizeA)

print("\nSubtask 14\n")
# Perform matrix multiplication of F and A if dimensions are compatible
if sizeF[1] == sizeA[0]:
    H = F @ A
    print("H:\n", H)
else:
    print("Cannot multiply F and A due to incompatible dimensions.")

print("\nSubtask 15\n")
# Generate the identity matrix with 3 rows and 3 columns
eye33 = np.eye(3)
print("eye33:\n", eye33)

print("\nSubtask 16\n")
# Generate matrices of zeros with size 5x3 and ones with size 4x2
zeros53 = np.zeros((5, 3))
ones42 = np.ones((4, 2))
print("zeros53:\n", zeros53)
print("ones42:\n", ones42)

print("\nSubtask 17\n")
# Generate a diagonal matrix S with the diagonal elements 1, 2, 7
S = np.diag([1, 2, 7])
print("S:\n", S)

print("\nSubtask 18\n")
# Extract the diagonal elements from a random 6x6 matrix
R = np.random.rand(6, 6)
diagR = np.diag(R)
print("R:", R)
print("diagR:", diagR)

print("\nSubtask 19\n")
# Create a sparse diagonal matrix and convert to dense
from scipy.sparse import diags
diag121 = diags([-np.ones(10), 2*np.ones(10), -np.ones(10)], [-1, 0, 1], shape=(10,
10)).todense()
print("diag121:\n", diag121)
