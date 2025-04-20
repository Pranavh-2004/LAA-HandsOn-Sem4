import numpy as np
from scipy.io import loadmat

# Project 5: Systems of linear equations and college football team ranking (with an example of the Big 12)

print("\nSubtask 1\n")
# Load the matrices from .mat files
data_scores = loadmat("/Users/pranavhemanth/Code/Academics/LA-S4/Scores.mat")
data_differentials = loadmat("/Users/pranavhemanth/Code/Academics/LA-S4/Differentials.mat")
# Extract the relevant matrices
Scores = data_scores['Scores']
Differentials = data_differentials['Differentials']

print("\nSubtask 2\n")
# Colley's method
# Define variables
games = np.abs(Scores)
total = np.sum(games, axis=1)
# Construct Colley's matrix and the right-hand side vector
ColleyMatrix = 2 * np.eye(10) + np.diag(total) - games
RightSide = 1 + 0.5 * np.sum(Scores, axis=1)
# Print to verify
print("ColleyMatrix:\n", ColleyMatrix)
print("\nRightSide:\n", RightSide)

print("\nSubtask 3\n")
# Solve the linear system using np.linalg.solve
RanksColley = np.linalg.solve(ColleyMatrix, RightSide)
# Variables: RanksColley
print("RanksColley:\n", RanksColley)

print("\nSubtask 4\n")
# Teams list
Teams = [
'Baylor', 'Iowa State', 'University of Kansas', 'Kansas State',
'University of Oklahoma', 'Oklahoma State', 'Texas Christian',
'University of Texas Austin', 'Texas Tech', 'West Virginia'
]
# Sort the ranks in descending order and get the order of indices
Order = np.argsort(RanksColley)[::-1]
RanksDescend = RanksColley[Order]
# Display the results
print('\n')
for j in range(10):
    print(f'{RanksColley[Order[j]]:8.3f} {Teams[Order[j]]:15s}')

print("\nSubtask 5\n")
# Massey's method
l = 0
P = []
B = []
# Loop through the upper triangular part of the Differentials matrix
for j in range(9):
    for k in range(j + 1, 10):
        if Differentials[j, k] != 0:
            l += 1
            row = np.zeros(10)
            row[j] = 1
            row[k] = -1
            P.append(row)
            B.append(Differentials[j, k])
# Convert lists to numpy arrays
P = np.array(P)
B = np.array(B)
# Variables: P, B
print("Matrix P:\n", P)
print("Vector B:\n", B)

print("\nSubtask 6\n")
# Create the normal system of linear equations
A = np.dot(P.T, P)
D = np.dot(P.T, B)
# Variables: A, D
print("Matrix A:\n", A)
print("Vector D:\n", D)

print("\nSubtask 7\n")
# Substitute the last row of the matrix and the last element of the vector
A[9, :] = np.ones(10)
D[9] = 0
# Print the updated matrix and vector
print("Updated Matrix A:\n", A)
print("Updated Vector D:\n", D)

print("\nSubtask 8\n")
# Solve the system
RanksMassey = np.linalg.solve(A, D)
# Print the results
print("RanksMassey:\n", RanksMassey)

print("\nSubtask 9\n")
# Teams list
Teams = ['Baylor', 'Iowa State', 'University of Kansas', 'Kansas State',
'University of Oklahoma', 'Oklahoma State', 'Texas Christian',
'University of Texas Austin', 'Texas Tech', 'West Virginia']
# Sort the ranks in descending order
Order = np.argsort(RanksMassey)[::-1]
RanksDescend = RanksMassey[Order]
# Print the results
print("\nMassey Rankings:")
for j in range(10):
    print(f'{RanksDescend[j]:8.3f} {Teams[Order[j]]:<15}')

print("\nSubtask 10\n")
print("Compare results of past 2 tasks")

print("\nSubtask 11\n")
# Identify the current top two teams according to Colley's rankings
top_teams_colley = sorted(range(len(RanksColley)), key=lambda i: RanksColley[i],
reverse=True)[:2]
# Simulate switching the result of the game between the top two teams
# For example, if team 0 (top_teams_colley[0]) played against team 1 (top_teams_colley[1])
# and team 0 lost, we switch it to a win.
Scores[top_teams_colley[0], top_teams_colley[1]] = -Scores[top_teams_colley[0],
top_teams_colley[1]]
# Recalculate Colley's rankings
games = np.abs(Scores)
total = np.sum(games, axis=1)
ColleyMatrix = 2 * np.eye(10) + np.diag(total) - games
RightSide = (1 + 0.5 * np.sum(Scores, axis=1))
RanksColley_updated = np.linalg.solve(ColleyMatrix, RightSide)
# Display the updated rankings
teams = ['Baylor', 'Iowa State', 'University of Kansas', 'Kansas State',
'University of Oklahoma', 'Oklahoma State', 'Texas Christian',
'University of Texas Austin', 'Texas Tech', 'West Virginia']
# Sort and print rankings
order_updated = np.argsort(RanksColley_updated)[::-1]
print("\nUpdated Colley's Rankings After Game Result Switch:")
for j in range(10):
    print(f"{RanksColley_updated[order_updated[j]]:8.3f} {teams[order_updated[j]]}")
# Reset the game result for future calculations
Scores[top_teams_colley[0], top_teams_colley[1]] = -Scores[top_teams_colley[0],
top_teams_colley[1]]

print("\nSubtask 12\n")
# Identify the current top two teams according to Massey's rankings
top_teams_massey = sorted(range(len(RanksMassey)), key=lambda i: RanksMassey[i],
reverse=True)[:2]
# Simulate switching the result of the game between the top two teams for Massey's method
# Adjust the differential matrix Differentials for the switched game result
# For example, if team 0 (top_teams_massey[0]) played against team 1
(top_teams_massey[1])
# and team 0 lost, we switch it to a win.
if Differentials[top_teams_massey[0], top_teams_massey[1]] != 0:
    Differentials[top_teams_massey[0], top_teams_massey[1]] = - Differentials[top_teams_massey[0], top_teams_massey[1]]
# Reset the initial matrix P to be large enough to accommodate all possible rows
P = np.zeros((45, 10))
B = np.zeros(45)
l = -1 # Initialize l to -1 because it will be incremented at the beginning of the loop
# Populate P and B based on the conditionals in your loop
for j in range(9):
    for k in range(j + 1, 10):
        if Differentials[j, k] != 0:
            l += 1
            P[l, j] = 1
            P[l, k] = -1
            B[l] = Differentials[j, k]
# Adjust for the last row substitution as described in the previous steps
P[44, :] = np.ones(10)
B[44] = 0
# Recalculate Massey's rankings
A = np.dot(P.T, P)
D = np.dot(P.T, B)
# Solve the system again
RanksMassey_updated = np.linalg.solve(A, D)
# Display the updated rankings
print("\nUpdated Massey's Rankings After Game Result Switch:")
order_updated_massey = np.argsort(RanksMassey_updated)[::-1]
for j in range(10):
    print(f"{RanksMassey_updated[order_updated_massey[j]]:8.3f} {teams[order_updated_massey[j]]}")
# Reset the game result for future calculations
if Differentials[top_teams_massey[0], top_teams_massey[1]] != 0:
    Differentials[top_teams_massey[0], top_teams_massey[1]] = - Differentials[top_teams_massey[0], top_teams_massey[1]]