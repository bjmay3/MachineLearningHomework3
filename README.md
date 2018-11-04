1. All coding done in Python.  Code files can be found at the following link:  https://github.com/bjmay3/MachineLearningHomework3

2. Twelve (12) separate Python code files exist.  These are as follows:
	a. KMeans_Car - K-Means Clustering on Car Evaluation dataset
	b. KMeans_Node - K-Means Clustering on Node Flooding dataset
	c. EM_Car - Expectation Maximization on Car Evaluation dataset
	d. EM_Node - Expectation Maximization on Node Flooding dataset
	e. PCA_Car - Principal Component Analysis on Car Evaluation dataset
	f. PCA_Node - Principal Component Analysis on Node Flooding dataset
	g. ICA_Car - Independent Component Analysis on Car Evaluation dataset
	h. ICA_Node - Independent Component Analysis on Node Flooding dataset
	i. RCA_Car - Random Componenet Analysis (Randomized Projections) on Car Evaluation dataset
	j. RCA_Node - Random Componenet Analysis (Randomized Projections) on Node Flooding dataset
	k. LDA_Car - Linear Discriminant Analysis on Car Evaluation dataset
	l. LDA_Node - Linear Discriminant Analysis on Node Flooding dataset

3. Each string of code could contain up to four (4) parts.  These are as follows:
	a. Part A - Data Preparation:  loads libraries, loads data, manipulates data as necessary.
	b. Part B - Run the model on the data.  Runs appropriate algorithm (K-Means, EM, PCA, ICA, RCA, or LDA).
	c. Part C - For K-Means_Node and EM_Node, uses clustering for dimensionality reduction and then runs an ANN on the dimensionally reduced data.
		  - For all PCA, ICA, RCA, and LDA files, uses the dimensionally reduced data to determine clusters
	d. Part D - For PCA, ICA, RCA, and LDA as they pertain to the Node Flooding dataset only, uses dimensionally reduced data for ANN classification.

4. At a minimum, run each part of the code separately.	The following results will be displayed by part:
	a. Data Preparation - various data sizes and data displays to indicate that download, transformation, and breaking into X and y components happened correctly.
	b. Model run on the data.
		- For K-Means and EM on the Car Evaluation dataset, produces graphs and downloaded .csv files used to conduct further analysis.
		- For K-Means and EM on the Node Flooding dataset, produces graphs of the clusters for various two-dimensional data arrangements.
		- For PCA, ICA, RCA, and LDA, produces dimensionally reduced datasets.
	c. Clustering as dimensionality reduction - transforms the data as a dimensionally reduced set of points with values equal to the euclidean distance from the cluster centers (means in the case of EM).  Once dimensionally reduced, the data is then run through the ANN classification algorithm.
	d. Clustering w/dimensionally reduced data - uses elbow graph to find optimal number of clusters then runs K-Means & EM algorithms, compares to actual results.
	e. ANN classification w/dimensionally reduced data - runs ANN classification on the Node Flooding dataset using data dimensionally reduced via the four methods (PCA, ICA, RCA, LDA); different numbers of hidden layers and numbers of nodes within each hidden layer can be explored.

5. Attribution:  Some code was "borrowed" from elsewhere.  The following gives credit to the places where various pieces of code were obtained.
	a. Udemy Super Data Science course - Data Preparation, predictions from test set, confusion matrix development, Decision Tree, KNN, SVM model training.
	b. Clustering as dimension reduction - question on Stack Exchange that refers to an excerpt from the book "Machine Learning With Spark" by Nick Pentreath.
	c. ANN code - ANN model training using MLPClassifier.

6. Attribution Websites
	a. Udemy Super Data Science course - https://www.udemy.com/datascience
	b. Clustering as dimension reduction - https://stats.stackexchange.com/questions/288668/clustering-as-dimensionality-reduction
	c. ANN code - https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html
		    - scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
