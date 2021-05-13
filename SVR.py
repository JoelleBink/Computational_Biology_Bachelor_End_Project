from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd


# load the data
train_data = pd.read_csv('CHEMBL239_dataset_train.csv')
test_data = pd.read_csv('CHEMBL239_dataset_test.csv')

#get x and y values
y_train = train_data.loc[:, 'exp_mean']
y_test = test_data.loc[:, 'exp_mean']
X_train = train_data.loc[:, 'Bit 1' : 'Bit 1024']
X_test = test_data.loc[:, 'Bit 1' : 'Bit 1024']

plt.scatter(x, y, s=5, color="blue")
plt.show()
