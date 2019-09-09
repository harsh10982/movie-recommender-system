from numpy import *
import scipy.optimize

# 10 movies
# 5 users
# 3 features

# Each user has preferences of movie features
# Each movie has features


# STEP: 1
# Set ratings data [col:User; row:Movie]
ratings = array([[8, 4, 0, 0, 4], [0, 0, 8, 10, 4], [8, 10, 0, 0, 6], [10, 10, 8, 10, 10], [0, 0, 0, 0, 0], [2, 0, 4, 0, 6], [8, 6, 4, 0, 0], [0, 0, 6, 4, 0], [0, 6, 0, 4, 10], [0, 4, 6, 8, 8]])


# Binary array for rated vs unrated movies
did_rate = (ratings != 0) * 1


# My movie ratings
my_ratings = zeros((10,1))
my_ratings[0] = 7
my_ratings[4] = 8
my_ratings[7] = 3


# Add my movie ratings to the matrices
ratings = append(my_ratings, ratings, axis = 1)
did_rate = append(((my_ratings!=0) * 1), did_rate, axis = 1)


# STEP: 2
# We must normalise the ratings so that the average rating is 0 for each movie
def normalise_ratings(ratings, did_rate):
  num_movies = ratings.shape[0]
  ratings_mean = zeros(shape = (num_movies, 1))
  ratings_norm = zeros(shape = ratings.shape)

  for i in range(num_movies):
    # Get all the indices where there is a rating
    index = where(did_rate[i] == 1)[0]

    # Calculate mean rating of the i'th movie only from users that gave a rating
    ratings_mean[i] = mean(ratings[i, index])
    ratings_norm[i, index] = ratings[i, index] - ratings_mean[i]

  return (ratings_norm, ratings_mean)


# Normalise ratings
ratings_norm, ratings_mean = normalise_ratings(ratings, did_rate)



# STEP: 3
# Use linear regression to find the best recommendations

# For figuring out the linear regression:
##  Theta is our vector of parameters (which will be user preferences to input when we use the linear regression later)
##  X is vector of actual features (movie features)



# Cost function
def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
  # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
  # --------------------------------------------------------------------------------------------------------------
  # Get the first 30 (10 * 3) rows in the 48 X 1 column vector
  first_30 = X_and_theta[:num_movies * num_features]
  # Reshape this column vector into a 10 X 3 matrix
  X = first_30.reshape((num_features, num_movies)).transpose()
  # Get the rest of the 18 the numbers, after the first 30
  last_18 = X_and_theta[num_movies * num_features:]
  # Reshape this column vector into a 6 X 3 matrix
  theta = last_18.reshape(num_features, num_users ).transpose()
  
  # we multiply by did_rate because we only want to consider observations for which a rating was given
  # we calculate the sum of squared errors here.  
  # in other words, we calculate the squared difference between our hypothesis (predictions) and ratings
  cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
  
  # we get the sum of the square of every element of X and theta
  regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
  return cost + regularization


# Calculate gradients, i.e. derivatives
def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
  # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
  # --------------------------------------------------------------------------------------------------------------
  # Get the first 30 (10 * 3) rows in the 48 X 1 column vector
  first_30 = X_and_theta[:num_movies * num_features]
  # Reshape this column vector into a 10 X 3 matrix
  X = first_30.reshape((num_features, num_movies)).transpose()
  # Get the rest of the 18 the numbers, after the first 30
  last_18 = X_and_theta[num_movies * num_features:]
  # Reshape this column vector into a 6 X 3 matrix
  theta = last_18.reshape(num_features, num_users ).transpose()
  
  # we multiply by did_rate because we only want to consider observations for which a rating was given
  difference = X.dot( theta.T ) * did_rate - ratings
  
  # we calculate the gradients (derivatives) of the cost with respect to X and theta
  X_grad = difference.dot( theta ) + reg_param * X
  theta_grad = difference.T.dot( X ) + reg_param * theta
  
  # wrap the gradients back into a column vector 
  return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# Generate sample data
num_movies, num_users = shape(ratings)
num_features = 3

# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = random.randn( num_movies, num_features )
user_prefs = random.randn( num_users, num_features )

# r_ is from numpy.r_ and creates an instance of a class which defines an array with the values in the squared brackets []
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]

# Regularization paramater
reg_param = 30.0

# STEP: 4
# Calculate the minimum cost and params used to achieve it

# fprime simply refers to the derivative (gradient) of the calculate_cost function
# We iterate 100 times
minimized_cost_and_optimal_params = scipy.optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, \
                args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), \
                maxiter=100, disp=True, full_output=True )


# Retrieve the minimized cost and the optimal values of the movie_features (X) and user_prefs (theta) matrices
cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


first_30 = optimal_movie_features_and_user_prefs[:num_movies * num_features]
movie_features = first_30.reshape((num_features, num_movies)).transpose()
last_18 = optimal_movie_features_and_user_prefs[num_movies * num_features:]
user_prefs = last_18.reshape(num_features, num_users ).transpose()

# Predictions for each user
all_predictions = movie_features.dot( user_prefs.T )

# My predictions
predictions_for_me = all_predictions[:, 0:1] + ratings_mean


# Get the movies from movies.txt into a Python dictionary
def loadMovies():
  movie_dict = {}
  movie_index = 0
  with open('./movies.txt', 'rb') as movieFile:
    file_contents = movieFile.readlines()
    for content in file_contents:
      print content
      print content.strip()
      print content.strip().split(' ',1)
      movie_dict[movie_index] = content.strip().split(' ', 1)[1]
      movie_index += 1

  return movie_dict

all_movies = loadMovies()


# STEP 5: Prepare our results

# Sort our predictions. We use argsort; we cannot simply use sort (predictions_for_me)
sorted_indexes = predictions_for_me.argsort(axis=0)[::-1]
predictions_for_me = predictions_for_me[sorted_indexes]
myPredictionsResult = []

# Display Predictions
# Since we only have 10 movies, let's display all ratings
for i in range(num_movies):
  # grab index (integer), which are all sorted based on prediction values
  index = sorted_indexes[i,0]
  print "Predicting rating %.1f for movie %s" % (predictions_for_me[index], all_movies[index])
  myPredictionsResult.append([predictions_for_me[index,0,0], all_movies[index]])



# Outputs
'''
print ratings
print did_rate
print ratings_norm
print ratings_mean

print 'all predictions'
print all_predictions
'''
for result in myPredictionsResult:
  print result



print 'We most recommend ' + max(myPredictionsResult)[1] 
print "with recommendation rating", max(myPredictionsResult)[0]