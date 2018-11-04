# Random Component Analysis (Car Evaluation)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA as RCA
from sklearn.cluster import KMeans
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

# Part B:  Run Random Component Analysis (RCA) algorithm

# Scale the independent variobles
sc = StandardScaler()
X = sc.fit_transform(X)

# Apply RCA to the independent variables
rca = RCA(random_state=1)
X = rca.fit_transform(X)
var = rca.explained_variance_ratio_
print(pd.DataFrame(var[:10]))

# Part C:  Use dimensionally reduced dataset to cluster

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

# Fitting K-Means to the dataset (4 clusters)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(X)

# Check against known classifications
cm = confusion_matrix(y_kmeans, Y)
print(Y_Results)
print (pd.DataFrame(cm))

# Run EM using GaussianMixture (4 clusters)
EM = GaussianMixture(n_components=4, random_state=1)
fit = EM.fit(X)
labels = fit.predict(X)

# Generate confusion matrix to compare to actual results
cm = confusion_matrix(labels, Y)
print(Y_Results)
print (pd.DataFrame(cm))
