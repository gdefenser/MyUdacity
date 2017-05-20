import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

""" Calculates and returns the performance score between 
    true and predicted values based on the metric chosen. """
def performance_metric(y_true, y_predict):
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    # Return the score
    return score

""" Performs grid search over the 'max_depth' parameter for a 
    decision tree regressor trained on the input data [X, y]. """
def fit_model(X, y):
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit()
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)
    # Create the grid search object
    grid = GridSearchCV(regressor, params)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_

data = pd.read_csv('/home/loveshadev/PycharmProjects/Udacity/ML_P5_PredictHousingPrice/bj_housing.csv')
prices = data['Value']
features = data.drop('Value', axis=1)

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])

# Produce a matrix for client data
client_data = [[128, 5, 1, 1, 2015, 20], # Client 1
               [166, 4, 2, 1, 2016, 1 ], # Client 2
               [100, 3, 2, 0, 2017, 10 ]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
   print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)

