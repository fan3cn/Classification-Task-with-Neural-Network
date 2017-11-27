import numpy as np
import tensorflow as tf
import os
import csv

train_data_file = "TrainData.csv"
cv_data_file = "CrossValidationData.csv"



def load_dataset(filename):
	X, Y = load_preprocessed_data(filename)
	return X.T, Y.T

# Load data from pre-processed training data
def load_preprocessed_data(filename):
	X = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1,usecols= range(1,506))
    # Size
	m,n = X.shape
	# Taining features
	X_train = X[:,:n-1]
	# Training labels
	Y_train = X[:,n-1:n]
	# Return
	return X_train, Y_train


