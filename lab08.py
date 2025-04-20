import numpy as np  
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.linalg import orth

# Project 8: Interpolation, extrapolation, and climate change

print("\nSubtask 1\n")
# High temperatures in Kansas (in Fahrenheit)
WeatherHigh = np.array([37, 44, 55, 66, 75, 84, 89, 88, 80, 69, 53, 41])
# Plot the temperatures
plt.figure()
plt.plot(range(1, 13), WeatherHigh, 'r-x')
plt.axis([1, 12, 30, 95])
plt.title('Average High Annual Temperatures in Kansas')
plt.xlabel('Month')
plt.ylabel('Temperature (F)')
plt.grid(True)
plt.show()

print("\nSubtask 2\n")
# Months: January, May, August, December
x = np.array([1, 5, 8, 12])
V = np.vander(x, increasing=True)
# Select corresponding temperatures
y = WeatherHigh[[0, 4, 7, 11]]
# Solve for polynomial coefficients
CoefHigh = np.linalg.lstsq(V, y, rcond=None)[0]
# Given months: January, May, August, and December
x = np.array([1, 5, 8, 12])
y = WeatherHigh[x-1] # Select corresponding temperatures
# Generate the Vandermonde matrix
V = np.vander(x, increasing=True)
# Solve for polynomial coefficients
CoefHigh = np.linalg.solve(V, y)
print("Vandermonde Matrix (V):")
print(V)
print("Coefficients of the cubic polynomial (CoefHigh):")
print(CoefHigh)
print("Coefficients of the cubic polynomial:", CoefHigh)

print("\nSubtask 3\n")
# Evaluate the polynomial at the given set of points
xc = np.arange(1, 12.1, 0.1)
ycHigh = np.polyval(CoefHigh[::-1], xc) # CoefHigh needs to be reversed for np.polyval
# Plot the polynomial and the original data
plt.figure()
plt.plot(xc, ycHigh, 'b-', label='Cubic Polynomial')
plt.plot(range(1, 13), WeatherHigh, 'r-x', label='Exact Data')
plt.axis([1, 12, 30, 95])
plt.xlabel('Month')
plt.ylabel('Temperature (High)')
plt.title('Polynomial Approximation of High Temperatures')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 4\n")
# Given months: January, March, May, August, October, and December
x_six = np.array([1, 3, 5, 8, 10, 12])
y_six = WeatherHigh[x_six - 1] # Select corresponding temperatures
# Generate the Vandermonde matrix
V_six = np.vander(x_six, increasing=True)
# Solve for polynomial coefficients
CoefHigh_six = np.linalg.solve(V_six, y_six)
print("Vandermonde Matrix for six months (V_six):")
print(V_six)
print("Coefficients of the 5th degree polynomial (CoefHigh_six):")
print(CoefHigh_six)
# Evaluate the polynomial at the given set of points
ycHigh_six = np.polyval(CoefHigh_six[::-1], xc) # CoefHigh_six needs to be reversed for
np.polyval
# Plot the polynomial and the original data
plt.figure()
plt.plot(xc, ycHigh_six, 'b-', label='5th Degree Polynomial')
plt.plot(range(1, 13), WeatherHigh, 'r-x', label='Exact Data')
plt.axis([1, 12, 30, 95])
plt.xlabel('Month')
plt.ylabel('Temperature (High)')
plt.title('5th Degree Polynomial Approximation of High Temperatures')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 5\n")
# All twelve months
x_all = np.arange(1, 13)
y_all = WeatherHigh
# Generate the Vandermonde matrix
V_all = np.vander(x_all, increasing=True)
# Solve for polynomial coefficients
CoefHigh_all = np.linalg.solve(V_all, y_all)
print("Vandermonde Matrix for all twelve months (V_all):")
print(V_all)
print("Coefficients of the 11th degree polynomial (CoefHigh_all):")
print(CoefHigh_all)
# Evaluate the polynomial at the given set of points
ycHigh_all = np.polyval(CoefHigh_all[::-1], xc) # CoefHigh_all needs to be reversed for
np.polyval
# Plot the polynomial and the original data
plt.figure()
plt.plot(xc, ycHigh_all, 'b-', label='11th Degree Polynomial')
plt.plot(range(1, 13), WeatherHigh, 'r-x', label='Exact Data')
plt.axis([1, 12, 30, 95])
plt.xlabel('Month')
plt.ylabel('Temperature (High)')
plt.title('11th Degree Polynomial Approximation of High Temperatures')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 6\n")
# Given data
x = np.arange(1, 13)
WeatherHigh = np.array([53, 60, 68, 77, 85, 90, 92, 89, 82, 71, 60, 55])
# Points for interpolation
xc = np.arange(1, 12.1, 0.1)
# Linear interpolation
ycHigh1 = np.interp(xc, x, WeatherHigh, left=None, right=None, period=None)
# Piecewise cubic Hermite interpolating polynomial (PCHIP)
from scipy.interpolate import PchipInterpolator
pchip_interpolator = PchipInterpolator(x, WeatherHigh)
ycHigh2 = pchip_interpolator(xc)
# Cubic spline interpolation
from scipy.interpolate import CubicSpline
spline_interpolator = CubicSpline(x, WeatherHigh)
ycHigh3 = spline_interpolator(xc)
# Plot the interpolations
plt.figure()
plt.plot(xc, ycHigh1, 'g-', label='Linear Interpolation')
plt.plot(xc, ycHigh2, 'r-', label='PCHIP Interpolation')
plt.plot(xc, ycHigh3, 'k-', label='Spline Interpolation')
plt.plot(x, WeatherHigh, 'bo', label='Original Data')
plt.axis([1, 12, 30, 95])
plt.xlabel('Month')
plt.ylabel('Temperature (High)')
plt.title('Comparison of Interpolation Methods')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 7\n")
# Load temperature data
data = loadmat("/Users/pranavhemanth/Code/Academics/LA-S4/temperature.mat")
temperature = data['temperature']

print("\nSubtask 8\n")
# Separate data into years and temperatures
years = temperature[:, 0]
temp = temperature[:, 1]
# Plot the temperature data
plt.figure()
plt.plot(years, temp, 'b-o')
plt.xlabel('Year')
plt.ylabel('Temperature (Celsius)')
plt.title('Average Yearly Temperatures')
plt.grid(True)
plt.show()

print("\nSubtask 9\n")
# Define the future years
futureyears = np.arange(2016, 2026)
# Linear extrapolation
linear_interpolator = interp1d(years, temp, kind='linear', fill_value='extrapolate')
futuretemp1 = linear_interpolator(futureyears)
# Piecewise cubic Hermite interpolating polynomial (PCHIP)
pchip_interpolator = PchipInterpolator(years, temp, extrapolate=True)
futuretemp2 = pchip_interpolator(futureyears)
# Cubic spline interpolation
spline_interpolator = CubicSpline(years, temp, extrapolate=True)
futuretemp3 = spline_interpolator(futureyears)
# Plot the extrapolated data
plt.figure()
plt.plot(years, temp, 'b-o', label='Original Data')
plt.plot(futureyears, futuretemp1, 'g-o', label='Linear Extrapolation')
plt.plot(futureyears, futuretemp2, 'r-x', label='PCHIP Extrapolation')
plt.plot(futureyears, futuretemp3, 'k-d', label='Spline Extrapolation')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.title('Extrapolation of Average Yearly Temperatures')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 10\n")
# Enter in command window : >> sum(temp)/n

print("\nSubtask 11\n")
# Find the orthagonal vectors

print("\nSubtask 12\n")
# Plot vectors in 11

print("\nSubtask 13\n")
print("Solution for subtasks 10-13\n")
# Calculate the average temperature
average_temp = np.mean(temp)
# Print the average temperature
print(f"The average temperature for the past 136 years was {average_temp:.4f}°C.")
# Calculate orthogonal projection
n = len(temp)
b1 = np.ones(n)
P1 = np.outer(b1, b1) / np.dot(b1, b1)
temp1 = P1 @ temp
# Print projection matrix and projected temperatures
print("Projection matrix P1:")
print(P1)
print("Projected temperatures (temp1):")
print(temp1)
# Plot the results
plt.figure()
plt.plot(years, temp, 'bo', label='Original Data')
plt.plot(years, temp1, 'g.', label='Projected Data')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Data Projection')
plt.legend()
plt.grid(True)
plt.show()

print("\nSubtask 14\n")
# Norm of P1^2 - P1
norm_P1 = np.linalg.norm(P1 @ P1 - P1)
print(f"norm(P1 * P1 - P1) = {norm_P1:.4e}")

print("\nSubtask 15\n")
# Create the matrix B2
m = len(years)
B2 = np.column_stack((np.ones(m), years))
# Create the orthonormal basis using orth
Q2 = orth(B2)
# Verify the ranks
rank_Q2 = np.linalg.matrix_rank(Q2)
rank_Q2_B2 = np.linalg.matrix_rank(np.column_stack((Q2, B2)))
print(f"Rank of Q2: {rank_Q2}")
print(f"Rank of [Q2, B2]: {rank_Q2_B2}")
# Q3: What kind of matrix is Q2^T Q2? Why?
Q2_T_Q2 = np.dot(Q2.T, Q2)
print("Q2.T @ Q2:")
print(Q2_T_Q2)

print("\nSubtask 16\n")
# Projection matrix onto the subspace S
P2 = Q2 @ Q2.T
# Project the temperature data
temp2 = P2 @ temp
# Plot the original and projected temperatures
plt.figure()
plt.plot(years, temp, 'bo', label='Original Data')
plt.plot(years, temp1, 'g.', label='Constant Approximation')
plt.plot(years, temp2, 'r.', label='Linear Approximation')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Data Projection')
plt.legend()
plt.grid(True)
plt.show()
# Display the projected temperatures
print("Projected temperatures (temp2):")
print(temp2)
# Norm of P2^2 - P2
norm_P2 = np.linalg.norm(P2 @ P2 - P2)
print(f"norm(P2 * P2 - P2) = {norm_P2:.4e}")

print("\nSubtask 17\n")
# Approximate using quadratic function

print("\nSubtask 18\n")
print("Combined solution for 17,18\n")
# Add a column of squared years to the matrix B3
B3 = np.column_stack((np.ones(m), years, years**2))
# Create the orthonormal basis Q3 using the orth function
Q3 = np.linalg.qr(B3)[0] # Using QR decomposition to get an orthonormal basis
# Projection matrix onto the quadratic subspace
P3 = Q3 @ Q3.T
# Project the temperature data onto the quadratic subspace
temp3 = P3 @ temp
# Plot the original and projected temperatures
plt.figure()
plt.plot(years, temp, 'bo', label='Original Data')
plt.plot(years, temp1, 'g.', label='Constant Approximation')
plt.plot(years, temp2, 'r.', label='Linear Approximation')
plt.plot(years, temp3, 'm.', label='Quadratic Approximation')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Data Projection')
plt.legend()
plt.grid(True)
plt.show()
# Display the projected temperatures
print("Projected temperatures (temp3):")
print(temp3)
# Norm of P3^2 - P3
norm_P3 = np.linalg.norm(P3 @ P3 - P3)
print(f"norm(P3 * P3 - P3) = {norm_P3:.4e}")

print("\nSubtask 19\n")
#19
# Define future years for prediction
futureyears = np.arange(2016, 2117)
# Perform spline interpolation and extrapolation for future temperatures
interp_func = interp1d(years, temp3, kind='cubic', fill_value='extrapolate')
futuretemp3 = interp_func(futureyears)
# Create a new figure for plotting
plt.figure()
# Plot future temperature predictions
plt.plot(futureyears, futuretemp3, 'g-')
plt.xlabel('Year')
plt.ylabel('Predicted Average Temperature (°C)')
plt.title('Predicted Average Temperature for Next 100 Years')
plt.grid()
# Display the predicted temperature for 2116
print(f'Predicted average temperature for the year 2116: {futuretemp3[-1]:.2f} °C')
plt.show()