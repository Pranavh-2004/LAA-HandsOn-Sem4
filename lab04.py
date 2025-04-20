import numpy as np
import time

# Project 4: Solving linear systems in Python

print("\nSubtask 1\n")
# Define a function to create a magic-like matrix
def create_magic_like(n):
    magic_square = np.zeros((n, n), dtype=int) 
    row, col = 0, n // 2
    num = 1
    while num <= n*n:
        magic_square[row, col] = num
        num += 1
        new_row, new_col = (row - 1) % n, (col + 1) % n
        if magic_square[new_row,new_col]:
            row += 1
        else:
            row, col = new_row, new_col
    return magic_square
# Generate a 5x5 "magic- like" matrix
A = create_magic_like(5)
# Define column vector b
b = np.array([10, 26, 42, 59,
38])
# Print matrix A and vector b
print("Matrix A:")
print(A)
print("\nVector b:", b)

print("\nSubtask 2\n")
# Solve the system Ax = b using numpy's solve function
x = np.linalg.solve(A, b)
# Print the solution vector x
print("Solution vector x:\n")
print(x)

print("\nSubtask 3\n")
# Calculate the residual r = A*x - b
r = np.dot(A, x) - b
# Print the residual vector r
print("\nResidual vector r:")
print(r)

print("\nSubtask 4\n")
import scipy.linalg
# Perform LU decomposition of A
P, L, U = scipy.linalg.lu(A) # Note: Use scipy.linalg.lu for LU decomposition
# Solve the system Ax = b using LU decomposition
x1 = np.linalg.solve(A, b)
# Calculate the error between solutions
err1 = np.dot(A, x1) - b
# Print matrices P, L, U and vectors x1, err1
print("Matrix P (Permutation matrix):")
print(P)
print("\nMatrix L (Lower triangular matrix):")
print(L)
print("\nMatrix U (Upper triangular matrix):")
print(U)
print("\nSolution vector x1:")
print(x1)
print("\nError vector err1:")
print(err1)

print("\nSubtask 5\n")
# Solve the system Ax = b using numpy's least squares function
y = np.linalg.lstsq(A, b, rcond=None)[0]
# Print the solution vector y
print("\nSolution vector y:")
print(y)

print("\nSubtask 6\n")
# Solve the system Ax = b using matrix inverse
A_inv = np.linalg.inv(A)
x2 = np.dot(A_inv, b)
# Calculate the residual r2 and error err2
r2 = np.dot(A, x2) - b
err2 = x - x2
# Print solution vector x2, residual r2 and error err2
print("\nSolution vector x2 using inverse:")
print(x2)
print("\nResidual vector r2:")
print(r2)
print("\nError vector err2:")
print(err2)

print("\nSubtask 7\n")
def rref(A):
    '''
    Computes the reduced row echelon
    form (RREF) of a matrix A.
    Parameters:
    A : numpy.ndarray
    Input matrix of shape (m, n)
    Returns:
    R : numpy.ndarray
    Reduced row echelon form of matrix
    A
    '''
    A = A.astype(float)
    m, n = A.shape
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        if A[r, lead] == 0:
            for i in range(r + 1, m):
                if A[i, lead] != 0:
                    A[[r, i]] = A[[i, r]]
                    break
        if A[r, lead] != 0:
            A[r] = A[r] / A[r, lead]
        for i in range(m):
            if i != r:
                A[i] -= A[i, lead] * A[r]
        lead += 1
    return A
# Compute reduced row echelon form of [A | b]
C = np.hstack((A, b[:, np.newaxis]))
R = rref(C)
# Extract solution vector x3 from the RREF matrix
x3 = R[:, -1]
# Calculate residuals
r3 = np.dot(A, x3) - b
err3 = x - x3
print("\nSolution vector x3 (from RREF):")
print(x3)
print("\nResidual vector r3 (from RREF):")
print(r3) 
print("\nError vector err3 (differencebetween x and x3):")
print(err3)

print("\nSubtask 8\n")
# Initialize Num
Num = 500
# Generate matrix A and vector b
A = np.random.rand(Num, Num) + Num * np.eye(Num)
b = np.random.rand(Num, 1)
# Method 1: Solve using backslash operator
start_time = time.time()
x1 = np.linalg.solve(A, b)
end_time = time.time()
time_backslash = end_time - start_time
# Method 2: Solve using matrix inverse
start_time = time.time()
x2 = np.linalg.inv(A) @ b
end_time = time.time()
time_inv = end_time - start_time
# Method 3: Solve using reduced row echelon form (rref)
start_time = time.time()
C = np.hstack((A, b))
R = rref(C)
x3 = R[:, -1]
end_time = time.time()
time_rref = end_time - start_time
# Print the solution vectors and computational times
print("\nSolution vector x1 (using backslash operator):")
print(x1)
print("\nSolution vector x2 (using matrix inverse):")
print(x2)
print("\nSolution vector x3 (using reduced row echelon form):")
print(x3)
print("\nComputational times:")
print(f"Backslash operator: {time_backslash} seconds")
print(f"Matrix inverse: {time_inv} seconds")
print(f"Reduced row echelon form (rref): {time_rref} seconds")

print("\nSubtask 9\n")
# Define matrix A and vector b forthe overdetermined system
A = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 10],
                [9, 11, 12]])

b = np.array([1, 2, 3, 4]).reshape(-1, 1) # Reshape b to be a columnvector
# Solve the system Ax = b using the backslash operator
x = np.linalg.lstsq(A, b,
rcond=None)[0]
# Calculate the residual r = Ax - b
r = np.dot(A, x) - b
# Print the solution vector x
print("\nSolution vector x:")
print(x)
# Print the residual vector r
print("\nResidual vector r:")
print(r)
# Calculate and print the solution using normal equations for comparison
ATA_inv = np.linalg.inv(np.dot(A.T,A)) # Calculate (A.T * A)^(-1)
ATb = np.dot(A.T, b) #Calculate A.T * b
y = np.dot(ATA_inv, ATb) #Solve normal equations A.T * A * y= A.T * b
# Calculate the difference between x and y to verify accuracy
err = x - y
# Print the solution vector y from normal equations
print("\nSolution vector y (from normal equations):")
print(y)
# Print the difference vector err (should be close to zero)
print("\nDifference vector err (x - y):")
print(err)

print("\nSubtask 10\n")
# Define matrix A and vector b for the underdetermined system
A = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9],
[10, 11, 12]])
b = np.array([1, 3, 5, 7]).reshape(-1, 1) #Remove the reshape, as it's unnecessary
# Solve the system Ax = b using the backslash operator
x = np.linalg.lstsq(A, b, rcond=None)[0]
# Calculate the residual r1 = Ax - b
r1 = np.dot(A, x) - b
# Print the solution vector x
print("\nSolution vector x:")
print(x)
# Print the residual vector r1
print("\nResidual vector r1:")
print(r1)
# Obtain another particular solution using the pseudoinverse pinv(A)
A_pinv = np.linalg.pinv(A)
y = np.dot(A_pinv, b)
# Calculate the residual r2 = Ax - b for solution y
r2 = np.dot(A, y) - b
# Print the solution vector y from pseudoinverse
print("\nSolution vector y (from pseudoinverse):")
print(y)
# Print the residual vector r2
print("\nResidual vector r2:")
print(r2)