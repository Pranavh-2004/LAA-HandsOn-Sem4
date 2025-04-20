'''DON'T CHANGE THE MAIN FUNCTION'''

import numpy as np

def gaussian_elimination(A, b):
    """
    Perform Gaussian elimination to solve Ax = b.

    Args:
        A (numpy.ndarray): Coefficient matrix of size n x n.
        b (numpy.ndarray): Column vector of size n.

    Returns:
        tuple: (U, x) where:
               U is the row echelon form of [A|b].
               x is the solution vector.
    """
    
    # your code here
    n = len(A)
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))  # Form the augmented matrix [A|b]
    
    # Forward Elimination: Convert to Row Echelon Form (without pivoting)
    for i in range(n):
        # --- Removed Partial Pivoting ---
        # Normalize the row to make the pivot element 1.
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        
        # Eliminate elements below the pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]
    
    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i + 1:n] * x[i + 1:n])) / augmented_matrix[i, i]
    
    return augmented_matrix[:, :-1], x
    # x = np.linalg.solve(A, b)
    # return (augmented_matrix[:, :-1], x)

# Boilerplate code to handle input and output
def main():
    # Read input
    n = int(input().strip())  # Read matrix size
    A, b = [], []

    for _ in range(n):
        row = list(map(float, input().split()))
        A.append(row[:-1])  # First n values go to A
        b.append(row[-1])  # Last value goes to b

    A = np.array(A)
    b = np.array(b)

    # Perform Gaussian elimination
    U, x = gaussian_elimination(A, b)

    # Print results
    print("U =")
    for row in U:
        print(" ".join(f"{x:.1f}" for x in row))

    print("\nx =")
    print(" ".join(f"{xi:.1f}" for xi in x))

if __name__ == "__main__":
    main()