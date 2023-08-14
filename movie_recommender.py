# Import libraries
import numpy as np
from scipy import optimize

# Define the number of movies and users
num_movies = 10
num_users = 5

# Randomly initialize movie ratings
ratings = np.random.randint(11, size=(num_movies, num_users))

print(ratings)

# Create a logical matrix for rated movies
did_rate = (ratings != 0).astype(int)

print(did_rate)

# Simulate user ratings
nikhil_ratings = np.zeros((num_movies, 1))
print(nikhil_ratings)

# Rate 3 movies
nikhil_ratings[0] = 8
nikhil_ratings[4] = 7
nikhil_ratings[7] = 3

print(nikhil_ratings)

# Update ratings and did_rate
ratings = np.hstack((nikhil_ratings, ratings))
did_rate = np.hstack(((nikhil_ratings != 0).astype(int), did_rate))

print(ratings)

# Normalize ratings
def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = np.zeros((num_movies, 1))
    ratings_norm = np.zeros_like(ratings)
    
    for i in range(num_movies):
        idx = np.where(did_rate[i] == 1)[0]
        ratings_mean[i] = np.mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

ratings, ratings_mean = normalize_ratings(ratings, did_rate)

# Update variables
num_users = ratings.shape[1]
num_features = 3

# Initialize movie features and user preferences
movie_features = np.random.randn(num_movies, num_features)
user_prefs = np.random.randn(num_users, num_features)
initial_X_and_theta = np.concatenate((movie_features.T.flatten(), user_prefs.T.flatten()))

# Unroll parameters
def unroll_params(X_and_theta, num_users, num_movies, num_features):
    first_30 = X_and_theta[:num_movies * num_features]
    X = first_30.reshape((num_features, num_movies)).T
    last_18 = X_and_theta[num_movies * num_features:]
    theta = last_18.reshape((num_features, num_users)).T
    return X, theta

# Calculate gradient
def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
    difference = X.dot(theta.T) * did_rate - ratings
    X_grad = difference.dot(theta) + reg_param * X
    theta_grad = difference.T.dot(X) + reg_param * theta
    return np.concatenate((X_grad.T.flatten(), theta_grad.T.flatten()))

# Calculate cost
def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
    cost = np.sum((X.dot(theta.T) * did_rate - ratings) ** 2) / 2
    regularization = (reg_param / 2) * (np.sum(theta**2) + np.sum(X**2))
    return cost + regularization

# Regularization parameter
reg_param = 30

# Perform gradient descent
minimized_cost_and_optimal_params = optimize.fmin_cg(
    calculate_cost, 
    fprime=calculate_gradient, 
    x0=initial_X_and_theta, 
    args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 
    maxiter=100, 
    disp=True, 
    full_output=True
)

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

# Unroll parameters again
movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

# Make predictions
all_predictions = movie_features.dot(user_prefs.T)

# Add back mean ratings
predictions_for_nikhil = all_predictions[:, 0:1] + ratings_mean

print(predictions_for_nikhil)
