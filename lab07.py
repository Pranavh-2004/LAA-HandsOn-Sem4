import numpy as np
import scipy.io  

# Project 7: Norms, angles, and your movie choices

print("\nSubtask 1\n")
# Load the .mat file
data = scipy.io.loadmat('users_movies.mat')
# Extract variables from the loaded data
movies = data['movies'] # Array of movie titles
users_movies = data['users_movies'] # Matrix of user ratings for movies
users_movies_sort = data['users_movies_sort'] # Extracted ratings for 20 most popular
movies
index_small = data['index_small'] # Indexes of the popular movies
trial_user = data['trial_user'] # Ratings of the popular movies by a trial user
# Get the dimensions of the users_movies matrix
m, n = users_movies.shape
# Print the variables and their dimensions to verify
print(f"Movies: {movies.shape}")
print(f"Users Movies: {users_movies.shape}")
print(f"Users Movies Sort: {users_movies_sort.shape}")
print(f"Index Small: {index_small.shape}")
print(f"Trial User: {trial_user.shape}")
print(f"Dimensions of users_movies: {m} rows, {n} columns")
# Variables: movies, users_movies, users_movies_sort, index_small, trial_user, m, n

print("\nSubtask 2\n")
# Print the titles of the 20 most popular movies
print('Rating is based on movies:')
# Loop through the index_small array and print the corresponding movie titles
for idx in index_small.flatten():
    print(movies[idx][0])
print('\n')

print("\nSubtask 3\n")
# Get the dimensions of the users_movies_sort matrix
m1, n1 = users_movies_sort.shape
# Initialize an empty list to store the ratings of users who have rated all 20 popular movies
ratings = []
# Loop through each row in users_movies_sort
for j in range(m1):
# Check if the product of the elements in the row is not zero (meaning no zeros in the row)
    if np.prod(users_movies_sort[j, :]) != 0:
# Append the row to the ratings list
        ratings.append(users_movies_sort[j, :])
# Convert the ratings list to a NumPy array
ratings = np.array(ratings)
# Print the resulting ratings array
print(f"Ratings: {ratings.shape}")

print("\nSubtask 4\n")
# Get the dimensions of the ratings matrix
m2, n2 = ratings.shape
# Initialize an empty list to store the Euclidean distances
eucl = []
# Loop through each row in ratings
for i in range(m2):
# Calculate the Euclidean distance between the trial_user vector and the current row of ratings
    distance = np.linalg.norm(ratings[i, :] - trial_user.flatten())
# Append the distance to the eucl list
eucl.append(distance)
# Convert the eucl list to a NumPy array
eucl = np.array(eucl)
# Print the resulting Euclidean distances
print(f"Euclidean distances: {eucl}")
# Variables: eucl

print("\nSubtask 5\n")
# Sort the Euclidean distances in ascending order
DistIndex = np.argsort(eucl)
MinDist = np.sort(eucl)
# Find the index of the closest user
closest_user_Dist = DistIndex[0]
# Print the results
print(f"Sorted Euclidean distances: {MinDist}")
print(f"Indices of users sorted by distance: {DistIndex}")
print(f"Index of closest user: {closest_user_Dist}")
# Variables: MinDist, DistIndex, closest_user_Dist

print("\nSubtask 6\n")
# Centralize the columns of the matrix ratings
ratings_cent = ratings - np.mean(ratings, axis=1).reshape(-1, 1)
# Centralize the trial_user vector
trial_user_cent = trial_user - np.mean(trial_user)
# Print the centralized ratings and trial_user vectors
print(f"Centralized ratings: \n{ratings_cent}")
print(f"Centralized trial_user: \n{trial_user_cent}")
# Variables: ratings_cent, trial_user_cent

print("\nSubtask 7\n")
# Initialize the pearson array
pearson = np.zeros(m2)
# Compute Pearson correlation coefficients
for i in range(m2):
    pearson[i] = np.corrcoef(ratings_cent[i, :], trial_user_cent.flatten())[0, 1]
# Print the resulting Pearson correlation coefficients
print(f"Pearson correlation coefficients: {pearson}")
# Variables: pearson

print("\nSubtask 8\n")
# Sort the Pearson correlation coefficients in descending order
PearsonIndex = np.argsort(pearson)[::-1]
MaxPearson = np.sort(pearson)[::-1]
# Find the index of the user with the highest correlation coefficient
closest_user_Pearson = PearsonIndex[0]
# Print the results
print(f"Sorted Pearson correlation coefficients:\n {MaxPearson}")
print(f"Indices of users sorted by Pearson correlation:\n {PearsonIndex}")
print(f"Index of user with highest Pearson correlation: {closest_user_Pearson}")
# Variables: MaxPearson, PearsonIndex, closest_user_Pearson

print("\nSubtask 9\n")
# Compare the elements of the vectors DistIndex and PearsonIndex
print("Indices sorted by Euclidean distance:", DistIndex)
print("Indices sorted by Pearson correlation:", PearsonIndex)
# Check if the variables closest_user_Pearson and closest_user_Dist are the same
if closest_user_Pearson == closest_user_Dist:
    print("The variables closest_user_Pearson and closest_user_Dist are the same.")
else:
    print("The variables closest_user_Pearson and closest_user_Dist are different.")

print("\nSubtask 10\n")
print("index_small shape:", index_small.shape)
print("trial_user shape:", trial_user.shape)
# Load the .mat file
data = scipy.io.loadmat('users_movies.mat')
# Extract variables from the loaded data
movies = data['movies'] # Array of movie titles
users_movies = data['users_movies'] # Matrix of user ratings for movies
users_movies_sort = data['users_movies_sort'] # Extracted ratings for 20 most popular
movies
index_small = data['index_small'].flatten() # Flatten index_small to 1D array
trial_user = data['trial_user'].flatten() # Ensure trial_user is 1D array
# Variables: movies, users_movies, users_movies_sort, index_small, trial_user
m, n = users_movies.shape
# Recommendations based on the distance criterion
recommend_dist = []
for k in range(n):
    if users_movies[closest_user_Dist, k] == 5:
        recommend_dist.append(k)
# Recommendations based on the Pearson correlation coefficient criterion
recommend_pearson = []
for k in range(n):
    if users_movies[closest_user_Pearson, k] == 5:
        recommend_pearson.append(k)
# Movies liked by the trial user
liked = []
for k in range(20):
    if trial_user[k] == 5:
# Convert 2D index_small to 1D index and add to liked list
        if k < len(index_small):
            liked.append(index_small[k])
# Convert indices to movie titles
liked_titles = [movies[i][0] for i in liked]
recommend_dist_titles = [movies[i][0] for i in recommend_dist]
recommend_pearson_titles = [movies[i][0] for i in recommend_pearson]
# Print the results
print("Movies liked by the trial user:", liked_titles)
print("Recommended movies based on distance criterion:", recommend_dist_titles)
print("Recommended movies based on Pearson correlation criterion:",
recommend_pearson_titles)
# Variables: liked, recommend_dist, recommend_pearson

print("\nSubtask 11\n")
# Function to print movie titles based on indices
def print_movie_titles(indices, movie_titles):
    print("Movie Titles:")
    for index in indices:
        print(movie_titles[index])
    print()
# Print titles of movies liked by the trial user
print("Movies liked by the trial user:")
print_movie_titles(liked, movies)
# Print recommendations based on the distance criterion
print("Recommended movies based on distance criterion:")
print_movie_titles(recommend_dist, movies)
# Print recommendations based on the Pearson correlation criterion
print("Recommended movies based on Pearson correlation criterion:")
print_movie_titles(recommend_pearson, movies)

print("\nSubtask 12\n")
# Manually specify your ratings for the 20 popular movies
# Example: Dislike some movies (1), Like some movies (5), Random ratings for others
# Replace these values with your own ratings
myratings = np.array([5, 1, 4, 3, 2, 5, 1, 4, 3, 5, 2, 5, 1, 3, 4, 5, 2, 1, 4, 3])
# Ensure myratings is a row vector (1D array with 20 elements)
print("My Ratings Vector:")
print(myratings)

print("\nSubtask 13\n")
# Load the .mat file
data = scipy.io.loadmat('users_movies.mat')
# Extract variables from the loaded data
movies = data['movies'] # Array of movie titles
users_movies = data['users_movies'] # Matrix of user ratings for movies
users_movies_sort = data['users_movies_sort'] # Extracted ratings for 20 most popular movies
index_small = data['index_small'].flatten() # Flatten index_small to 1D array
# Define your own ratings vector (myratings)
myratings = np.array([5, 1, 4, 3, 2, 5, 1, 4, 3, 5, 2, 5, 1, 3, 4, 5, 2, 1, 4, 3])
# Step 4: Select users who rated all 20 popular movies
[m1, n1] = users_movies_sort.shape
ratings = [users_movies_sort[j, :] for j in range(m1) if np.all(users_movies_sort[j, :] > 0)]
ratings = np.array(ratings)
# Step 5: Compute Euclidean distances
eucl = np.linalg.norm(ratings - myratings, axis=1)
# Step 6: Find the closest user based on Euclidean distance
MinDist, DistIndex = np.sort(eucl), np.argsort(eucl)
closest_user_Dist = DistIndex[0]
# Step 7: Centralize ratings and myratings for Pearson correlation
ratings_cent = ratings - np.mean(ratings, axis=1, keepdims=True)
myratings_cent = myratings - np.mean(myratings)
# Step 8: Compute Pearson correlation coefficients
pearson = np.sum(ratings_cent * myratings_cent, axis=1) / (
np.sqrt(np.sum(ratings_cent ** 2, axis=1)) * np.sqrt(np.sum(myratings_cent ** 2))
)
# Step 9: Find the closest user based on Pearson correlation
MaxPearson, PearsonIndex = np.sort(pearson)[::-1], np.argsort(pearson)[::-1]
closest_user_Pearson = PearsonIndex[0]
# Step 10: Create recommendations based on the distance criterion
recommend_dist = [k for k in range(users_movies.shape[1]) if users_movies[closest_user_Dist, k]
== 5]
# Step 11: Create recommendations based on the Pearson correlation criterion
recommend_pearson = [k for k in range(users_movies.shape[1]) if
users_movies[closest_user_Pearson, k] == 5]
# Create the list of movies liked by the trial user
liked = [index_small[k] for k in range(20) if myratings[k] == 5]
# Convert indices to movie titles
liked_titles = [movies[i][0] for i in liked]
recommend_dist_titles = [movies[i][0] for i in recommend_dist]
recommend_pearson_titles = [movies[i][0] for i in recommend_pearson]
# Print the results
print("Movies liked by the trial user:")
print(liked_titles)
print("Recommended movies based on distance criterion:")
print(recommend_dist_titles)
print("Recommended movies based on Pearson correlation criterion:")
print(recommend_pearson_titles)