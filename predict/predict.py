import numpy as np
import math

def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0,Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0,Z2)
    Z3 = np.dot(W3, A2) + b3
    
    return Z3

def predict(file1,file2,type_name, has_label):
    # Load parameters
    W1 = np.loadtxt("../train/parameters/W1.txt")
    b1 = np.asmatrix(np.loadtxt("../train/parameters/b1.txt")).T
    W2 = np.loadtxt("../train/parameters/W2.txt")
    b2 = np.asmatrix(np.loadtxt("../train/parameters/b2.txt")).T
    W3 = np.asmatrix(np.loadtxt("../train/parameters/W3.txt"))
    b3 = np.asmatrix(np.loadtxt("../train/parameters/b3.txt")).T

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}    
    # Load Test Data
    X_test = np.loadtxt(open(file1, "rb"), delimiter=",",usecols= range(1,505))
    # Forward propagation
    Z = forward_propagation(X_test.T, parameters)
    # Sigmoid
    Y = (sigmoid(Z) >= 0.5) * 1
    Y = Y.T
    # User Id
    X_user_1 = np.loadtxt(open(file1, "rb"), delimiter=",", usecols = 0, dtype='str') 
    # Output
    X_user_2 = np.loadtxt(open(file2, "rb"), delimiter=",", skiprows=1, usecols = 0, dtype='str')
    #predict = np.ones(X_user_unique.shape)
    for i in range(X_user_2.shape[0]):
        for j in range(X_user_1.shape[0]):
            if X_user_1[j] == X_user_2[i]:
                print(X_user_2[i]+","+str(Y[j,0])) 

predict("TestExtractedFeatures.csv", "../dataset/TestData.csv", "Test", 0)
#predict("../train/extracted_features.csv", "../dataset/TrainLabel.csv", "Test", 0)
