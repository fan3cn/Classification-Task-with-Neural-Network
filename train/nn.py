
# coding: utf-8

import math
import numpy as np
import tensorflow as tf
import load_data as ld
import random
from tensorflow.python.framework import ops



# GRADED FUNCTION: create_placeholders

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x):

    W1 = tf.get_variable("W1", [25, n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):

    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)             
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3

# GRADED FUNCTION: compute_cost 

def compute_cost(parameters, Z3, Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    # sigmoid_cross_entropy
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # L2 regulazation
    cost = cost + 0.01 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3))

    return cost


# 3 Layer Neural Network
def model(X_train, Y_train, X_cv, Y_cv, learning_rate = 0.001,
          num_epochs = 2000, minibatch_size = 64, print_cost = True):
    
    #ops.reset_default_graph()                        # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(parameters, Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.cast(tf.nn.sigmoid(Z3)>=0.5, "float"), Y)
        # Batch Size
        batch_size = tf.cast(tf.shape(Y)[1], "float")
        # Accuracy
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))/ batch_size
        # Training Accuracy
        print("Training Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # CrossValidation Accuracy
        print("CrossValidation Accuracy:", accuracy.eval({X: X_cv, Y: Y_cv}))
        
        return parameters

