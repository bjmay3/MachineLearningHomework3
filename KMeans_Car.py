# K-Means Clustering (Car Evaluation)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
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

# Part B:  Run K-Means Clustering algorithm

# Fitting K-Means to the dataset (4 clusters)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(X)

# Check against known classifications
cm = confusion_matrix(y_kmeans, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 19], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 19], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 19], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 19], s = 100, c = 'magenta', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 19], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters Visualized')
plt.xlabel(X_Headers[0])
plt.ylabel(X_Headers[19])
plt.legend()
plt.show()

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('Finding the Best K:  The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset (K = 9)
kmeans = KMeans(n_clusters = 9, init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(X)

# Build a matrix to compare clusters with variables and export it
dataset1 = pd.read_csv('CarRatingDataset.csv')
dataset1['Cluster'] = y_kmeans
dataset1.to_csv('ouput', header=True, sep=',')
