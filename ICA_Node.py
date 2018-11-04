# Independent Component Analysis (Node Flooding)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
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

# Part B:  Run Independent Component Analysis (ICA) algorithm

# Scale the independent variobles
sc = StandardScaler()
X = sc.fit_transform(X)

# Apply ICA to the independent variables
ica = FastICA(n_components = None, random_state=1)
X_new = ica.fit_transform(X)

# Part C:  Use dimensionally reduced dataset to cluster

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
    kmeans.fit(X_new)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('Finding the Best K:  The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset (4 clusters)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(X_new)

# Check against known classifications
cm = confusion_matrix(y_kmeans, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Run EM using GaussianMixture (4 clusters)
EM = GaussianMixture(n_components=4, random_state=1)
fit = EM.fit(X_new)
labels = fit.predict(X_new)

# Generate confusion matrix and compare to actual results
cm = confusion_matrix(labels, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Part D:  Use dimensionality reduction results on ANN

# Use three compoonents and recompute ICA; apply to data
ica = FastICA(n_components = 3, random_state = 1)
X_new = ica.fit_transform(X)

# Redo the ANN classifier on the dataset & cross-validate
# 3 hidden layers each with 24 nodes per layer
classifier = MLPClassifier(hidden_layer_sizes = (24, 24, 24), random_state=1)
classifier.fit(X_new, Y)
scores = cross_val_score(classifier, X_new, Y, cv=4)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with around 46.5% accuracy

# Use all compoonents and recompute ICA; apply to data
ica = FastICA(n_components = None, random_state = 1)
X_new = ica.fit_transform(X)

# Redo the ANN classifier on the dataset & cross-validate
# 2 hidden layers each with 24 nodes per layer
classifier = MLPClassifier(hidden_layer_sizes = (24, 24), max_iter = 500,  random_state=1)
classifier.fit(X_new, Y)
scores = cross_val_score(classifier, X_new, Y, cv=4)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with around 80% accuracy
