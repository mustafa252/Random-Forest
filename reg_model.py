# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.datasets import load_diabetes


# load data 'penguins'
diabetes = load_diabetes()
diabetes.keys()


x = diabetes.data
y = diabetes.target

#####################################################################################
######### data analysis



####################################################################################
######### data visualization



####################################################################################
######### split data


X_train, X_test, Y_train, Y_test = train_test_split(x,y,
                                                    test_size=0.3,
                                                    random_state=42)

############################################################################################
############ Feature Scaling

from sklearn.preprocessing import StandardScaler

# standardisation
scaler = StandardScaler()
# apply scaler
X_train = scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)

####################################################################################
######## training & prediction

from sklearn.ensemble import RandomForestRegressor

# Regression
Regression = RandomForestRegressor(random_state=42)

# fit
Regression.fit(X_train, Y_train)

# predict
y_pred = Regression.predict(X_test)


 ####################################################################################
######## evaluation


from sklearn.metrics import mean_squared_error, r2_score

#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(Y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(Y_test, y_pred))






############################################################################################
############ hyperparameter tuninig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# hyperparameters set
params = {'n_estimators': [100, 200, 400],
          'max_depth':[10, 20, 30],
          'min_samples_split':[2,10,100],
          'min_samples_leaf':[1,5,10]}


# esclating the training by using all the cpu's
Regression = RandomForestRegressor(n_jobs=-1)
# Grid Search
grid = GridSearchCV(Regression,
                    param_grid=params,
                    cv=5,
                    n_jobs=-1)

grid.fit(X_train, Y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score
#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(Y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(Y_test, y_pred))



# Random Search
Random = RandomizedSearchCV(Regression,
                    params,
                    cv=5,
                    n_jobs=-1)

Random.fit(X_train, Y_train)


# show the best set
Random.best_estimator_
Random.best_params_
Random.best_score_


# predict
y_pred = Random.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score
#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(Y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(Y_test, y_pred))
