# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:24:00 2017
@author: richa
"""
import numpy as np
from scipy.spatial import distance
from collections import Counter

class KNNClassifier(object):
    """K-nearest neighbor classifier.
    """
    def __init__(self, n_neighbors=3, metric=distance.euclidean):
        """Initialize the kNN object.
        
        Args:
            n_neighbors (int, optional): number of neighbors to use by default 
            for KNN (default = 3)
        
            metric (callable, optional): the metric used to calculate the 
            distance between two arrays (default = distance.euclidean)    
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        if not callable(self.metric):
            raise TypeError("metric must be callable")
    
    def _cdist(self, X, Y):
        """Computes distance between each pair of the two collections of inputs
        
        Args:
            X (list or array): array with length of mX.
            Y (list or array): array with length of mY.
        
        Returns:
            dm (array): A distance matric, D. For each (i,j) element, the 
            metric(X_i, Y_j) is computed and stored in the distance matric.
        """
        X = np.array(X)
        Y = np.array(Y)
        mX = X.shape[0]
        mY = Y.shape[0]
        
        # Zero matric for starting
        dm = np.zeros((mX, mY), dtype=np.double)     
        
        # Get the metric function
        metric = self.metric
        
        # Calculate the pairwise distance
        for i in range(0, mX):
            for j in range(0, mY):
                dm[i, j] = metric(X[i], Y[j])
                    
        return dm
    
    def fit(self, training_data, training_label):
        """Fit the model by training data (training_data) and label 
        (training_label) 
        
        Args:
            training_data (list or array): the length of list or array should 
            equal the number of data points.
            training_label (list or array): the length of list or array should 
            equal the number of data points.
        """
        # check the dimension of training data and label
        training_data = np.array(training_data)
        training_label = np.array(training_label)
        data_samples = training_data.shape[0]
        label_samples = training_label.shape[0]
        
        if data_samples != label_samples:
            raise ValueError("Data and label samples must have same size.")
        if data_samples < self.n_neighbors:
            raise ValueError("Data size must be greater than n_neighbors.")
        if data_samples == 0:
            raise ValueError("Data size must be greater than 0.")
        
        # store the data and label for future training
        self.training_data = np.array(training_data)
        self.training_label = np.array(training_label).reshape(-1,)
        
    def _get_KNN_labels(self, d_matrix, n_neighbors):
        
        # Get the indices that would sort an array.
        sorted_indices = d_matrix.argsort()
        
        # Get the indices for the n nearest inputs
        KNN_indices = sorted_indices[:,:n_neighbors]
        
        # Get the k nearest labels
        KNN_labels = self.training_label[KNN_indices]
        
        return KNN_labels
    
    def predict(self, testing_data):
        """Predict the labeling for the testing data.
        Args:
            testing_data (list or array): length of list or array equal the 
            number of testing data points.
          
        Returns:
            array: the predicted label
        """
        testing_data = np.array(testing_data)
        dm = self._cdist(testing_data, self.training_data)
        KNN_labels = self._get_KNN_labels(dm, self.n_neighbors)
        
        voting_statistics = [Counter(KNN_label).most_common() for KNN_label in KNN_labels]
        predicted_label = [vote[0][0] for vote in voting_statistics]
        
        return np.array(predicted_label)
        
if __name__ == "__main__":
    clf = KNNClassifier(n_neighbors=3, metric=distance.euclidean)
    print( clf.__class__.__name__)