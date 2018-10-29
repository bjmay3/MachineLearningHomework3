# Expectation Maximization (Node Flooding)

# K-Means Clustering (Node Flooding)

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

# Run EM using GaussianMixture (4 clusters)
EM = GaussianMixture(n_components=4, random_state=33)
fit = EM.fit(X)
labels = fit.predict(X)

# Generate confusion matrix and compare to actual results
cm = confusion_matrix(labels, Y)
print(Y_Results)
print (cm)

# Run EM using GaussianMixture (3 clusters)
EM = GaussianMixture(n_components=3, random_state=33)
fit = EM.fit(X)
labels = fit.predict(X)

# Visualising the clusters
plt.scatter(X[labels == 0, 4], X[labels == 0, 5], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[labels == 1, 4], X[labels == 1, 5], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[labels == 2, 4], X[labels == 2, 5], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(EM.means_[:, 4], EM.means_[:, 5], s = 300, c = 'yellow', label = 'Means')
plt.title('Clusters Visualized')
plt.xlabel(X_Headers[4])
plt.ylabel(X_Headers[5])
plt.legend()
plt.show()
