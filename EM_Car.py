# Expectation Maximization (Car Evaluation)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

# Set the working directory (set to directory containing the dataset)
os.chdir('C:\\Users\Brad\Desktop\Briefcase\Personal\GeorgiaTechMasters\CS7641_MachineLearning\Homework\Homework3')

# Importing the dataset
dataset = pd.read_csv('CarRatingDataset.csv')
print ("Dataset Length = ", len(dataset))
print ("Dataset Shape = ", dataset.shape)
dataset.head()

# Break up the dataset into X and Y components
X = dataset.iloc[:, :6].values
Y = dataset.iloc[:, 6].values
print(X[:10, :])
print(Y[:10])

# Encode the categorical data
# Encode the Independent variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 4, 5])
X = onehotencoder.fit_transform(X).toarray()
X_Headers = np.array(['BuyPrice_high', 'BuyPrice_low', 'BuyPrice_med',
                      'BuyPrice_vhigh', 'MaintPrice_high', 'MaintPrice_low',
                      'MaintPrice_med', 'MaintPrice_vhigh', '2-door', '3-door',
                      '4-door', '5more-door', '2-pass', '4-pass', '5more-pass',
                      'Luggage_big', 'Luggage_med', 'Luggage_small',
                      'safety_high', 'safety_low', 'safety_med'])
print(X_Headers)
print(X[:10, :])

# Encode the dependent variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
Y_Results = np.array(['0=acc', '1=good', '2=unacc', '3=vgood'])
print(Y_Results)
print(Y[:10])

# Part B:  Run Expectation Maximization (EM) algorithm

# Run EM using GaussianMixture (4 clusters)
EM = GaussianMixture(n_components=4, random_state=1)
fit = EM.fit(X)
labels = fit.predict(X)

# Generate confusion matrix to compare to actual results
cm = confusion_matrix(labels, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Visualising the clusters
plt.scatter(X[labels == 0, 0], X[labels == 0, 19], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 19], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 19], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[labels == 3, 0], X[labels == 3, 19], s = 100, c = 'magenta', label = 'Cluster 4')
plt.scatter(EM.means_[:, 0], EM.means_[:, 19], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters Visualized')
plt.xlabel(X_Headers[0])
plt.ylabel(X_Headers[19])
plt.legend()
plt.show()

# Run EM using GaussianMixture (9 clusters)
EM = GaussianMixture(n_components=9, random_state=1)
fit = EM.fit(X)
labels = fit.predict(X)

# Build a matrix to compare clusters with variables and export it
dataset1 = pd.read_csv('CarRatingDataset.csv')
dataset1['Cluster'] = labels
dataset1.to_csv('ouput', header=True, sep=',')
