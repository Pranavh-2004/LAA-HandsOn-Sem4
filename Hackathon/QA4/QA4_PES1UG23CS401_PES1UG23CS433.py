'''DON'T CHANGE THE MAIN FUNCTION'''

import numpy as np

def analyze_sensor_matrix(matrix):
	"""
	Determines whether the sensor matrix is singular.
	If non-singular, computes its inverse or pseudo-inverse.

	Args:
		matrix (numpy.ndarray): The sensor data matrix.

	Returns:
		tuple: (is_singular, inverse matrix or None)
	"""

	if matrix.shape[0] == matrix.shape[1]:
		try:
			inverse = np.linalg.inv(matrix)
			return (False, inverse)

		except np.linalg.LinAlgError as err:
			return (True, None)

	else:
		try:
			inverse = np.linalg.pinv(matrix)
			return (False, inverse)

		except np.linalg.LinAlgError as err:
			return (False, None)

def main():
	# Read input
	n = int(input())
	A = []
	for _ in range(n):
		row = list(map(float, input().split()))
		A.append(row)

	# Convert to NumPy array
	A = np.array(A)

	# Analyze the sensor matrix
	is_singular, inv_matrix = analyze_sensor_matrix(A)

	# Print results
	if is_singular:
		print("The matrix is singular.")
	else:
		print("The matrix is not singular.")
		print("Inverse of the matrix:")
		print(np.round(inv_matrix, 2))

if __name__ == "__main__":
	main()
