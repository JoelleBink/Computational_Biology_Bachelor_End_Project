from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, max_error, r2_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# load the data
train_data = pd.read_csv('CHEMBL239_dataset_train.csv')
test_data = pd.read_csv('CHEMBL239_dataset_test.csv')

#get x and y values
y_train = train_data.loc[:, 'exp_mean']
y_test = test_data.loc[:, 'exp_mean']
X_train = train_data.loc[:, 'Bit 1' : 'Bit 1024']
X_test = test_data.loc[:, 'Bit 1' : 'Bit 1024']

variance_score = []
max_error_score = []
r2_value = []
neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in tqdm(range(1, 10)):
    # create model
    knn = KNeighborsRegressor(n_neighbors=i, n_jobs=-1)

    # fit the data
    knn.fit(X_train,y_train)

    # make prediction
    y_pred_knn=knn.predict(X_test)

    # compute regression metrics
    variance_score.append(explained_variance_score(y_test, y_pred_knn))
    max_error_score.append(max_error(y_test, y_pred_knn))
    r2_value.append(r2_score(y_test, y_pred_knn))

# Create figures
fig, axes = plt.subplots(1, 3,  figsize=(20, 5))

# plot figures
axes[0].plot(neighbors, variance_score, label = 'variance_score')
axes[1].plot(neighbors, max_error_score, label = 'max_error_score')
axes[2].plot(neighbors, r2_value, label = 'r2_score')

# Set axes for the explained_variance_score figure
for i in range(3):
    axes[i].set_ylabel('score')
    axes[i].set_xlabel('neighbors')
    axes[i].legend()

# To adjust the spacing between plots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

# Set the titles for both figures
axes[0].set_title("The explained_variance_score \n metric for KNN regression")
axes[1].set_title("The max_error_score metric \n for KNN regression")
axes[2].set_title("The r2_score metric \n for KNN regression")
