# K-Means Clustering (Node Flooding)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

# Set the working directory (set to directory containing the dataset)
os.chdir('C:\\Users\Brad\Desktop\Briefcase\Personal\GeorgiaTechMasters\CS7641_MachineLearning\Homework\Homework3')

# Importing the dataset
dataset = pd.read_csv('NodeFlooding.csv')
print ("Dataset Length = ", len(dataset))
print ("Dataset Shape = ", dataset.shape)
dataset.head()

# Break up the dataset into X and Y components
X = dataset.iloc[:, [19, 1, 2, 4, 6, 7, 16, 17, 18, 20]].values
Y = dataset.iloc[:, 21].values
print(X[:10, :])
print(Y[:10])

# Encode the categorical data
# Encode the Independent variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X_Headers = np.array(['NodeStatus_B', 'NodeStatus_NB', 'NodeStatus_PNB',
                      'UtilBandwRate', 'PacketDropRate', 'AvgDelay_perSec',
                      'PctLostByteRate', 'PacketRecRate', '10RunAvgDropRate',
                      '10RunAvgBandwUse', '10RunDelay', 'FloodStatus'])
print(X_Headers)
print(X[:10, :])

# Encode the dependent variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
Y_Results = np.array(['0=Block', '1=NB-No_Block', '2=NB-Wait', '3=No_Block'])
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

# Fitting K-Means to the dataset (3 clusters)
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters (3/4, 5/6, 9/11)
plt.scatter(X[y_kmeans == 0, 9], X[y_kmeans == 0, 11], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 9], X[y_kmeans == 1, 11], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 9], X[y_kmeans == 2, 11], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 9], kmeans.cluster_centers_[:, 11], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters Visualized')
plt.xlabel(X_Headers[9])
plt.ylabel(X_Headers[11])
plt.legend()
plt.show()

# Part C:  Using K-Means clustering as dimension reduction

# Determine centers of the clusters (using 3 clusters from before)
centers = kmeans.cluster_centers_

# Dimensionally reduced matrix represents euclidean distances between
# points in original dataset and each of the centers
X_new = np.empty((1075, 3), dtype=float)
for j in range(0, len(centers)):
    for i in range(0, len(X)):
        X_new[i, j] = distance.euclidean(X[i], centers[j])

# Scale the new results
sc = StandardScaler()
X_new = sc.fit_transform(X_new)

# Redo the ANN classifier on the dataset & cross-validate
# 2 hidden layers each with 24 nodes per layer
classifier = MLPClassifier(hidden_layer_sizes = (24, 24), max_iter = 200, random_state=1)
classifier.fit(X_new, Y)
scores = cross_val_score(classifier, X_new, Y, cv=4)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with around 71% accuracy
    