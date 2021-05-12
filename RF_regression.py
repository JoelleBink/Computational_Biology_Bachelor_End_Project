from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, max_error, r2_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# load the data and create test and validation set
train_data = pd.read_csv('CHEMBL239_dataset_train.csv')
test_data = pd.read_csv('CHEMBL239_dataset_test.csv')

#get x and y values
y_train = train_data.loc[:, 'exp_mean']
y_test = test_data.loc[:, 'exp_mean']
X_train = train_data.loc[:, 'Bit 1' : 'Bit 1024']
X_test = test_data.loc[:, 'Bit 1' : 'Bit 1024']

# Set the parameters to check with GridSearch
param = {'n_estimators': [50, 100, 500, 1000]}

# GridSearch on RandomForestRegressor
rf_model = GridSearchCV(RandomForestRegressor(), param, cv=2, return_train_score=True, n_jobs=-1, verbose=1).fit(X_train,y_train)

# store the best n_estimator value
best_param = rf_model.best_params_['n_estimators']

# make randomforest with best value for n_estimator
best_rf_model = RandomForestRegressor(n_estimators = best_param).fit(X_train,y_train)

# make prediction
y_pred_rf=best_rf_model.predict(X_test)

# print metrics for RandomForest
print('The best value of n_estimators is ', best_param)
print('\n')
print('variance_score:', explained_variance_score(y_test, y_pred_rf))
print('max_error_score:', max_error(y_test, y_pred_rf))
print('r2_value:', r2_score(y_test, y_pred_rf))