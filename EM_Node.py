# Expectation Maximization (Node Flooding)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
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

# Part B:  Run Expectation Maximization (EM) algorithm

# Run EM using GaussianMixture (4 clusters)
EM = GaussianMixture(n_components=4, random_state=1)
fit = EM.fit(X)
labels = fit.predict(X)

# Generate confusion matrix and compare to actual results
cm = confusion_matrix(labels, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Run EM using GaussianMixture (3 clusters)
EM = GaussianMixture(n_components=3, random_state=1)
fit = EM.fit(X)
labels = fit.predict(X)

# Visualising the clusters (3/4, 5/6, 9/11)
plt.scatter(X[labels == 0, 9], X[labels == 0, 11], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[labels == 1, 9], X[labels == 1, 11], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[labels == 2, 9], X[labels == 2, 11], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(EM.means_[:, 9], EM.means_[:, 11], s = 300, c = 'yellow', label = 'Means')
plt.title('Clusters Visualized')
plt.xlabel(X_Headers[9])
plt.ylabel(X_Headers[11])
plt.legend()
plt.show()

# Part C:  Using EM clustering as dimension reduction

# Determine centers (means) of the clusters (using 3 clusters from before)
centers = EM.means_

# Dimensionally reduced matrix represents euclidean distances between
# points in original dataset and each of the centers (means)
X_new = np.empty((1075, 3), dtype=float)
for j in range(0, len(centers)):
    for i in range(0, len(X)):
        X_new[i, j] = distance.euclidean(X[i], centers[j])

# Scale the new results
sc = StandardScaler()
X_new = sc.fit_transform(X_new)

# Redo the ANN classifier on the dataset & cross-validate
# 3 hidden layers each with 24 nodes per layer
classifier = MLPClassifier(hidden_layer_sizes = (24, 24), random_state=1)
classifier.fit(X_new, Y)
scores = cross_val_score(classifier, X_new, Y, cv=4)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with around 71% accuracy
