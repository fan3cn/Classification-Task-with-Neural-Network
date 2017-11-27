import load_data as ld
import nn
import numpy as np

# Run the following cell to train your model! On our machine it takes about 5 minutes. Your "Cost after epoch 100" should be 1.016458. If it's not, don't waste time; interrupt the training by clicking on the square (⬛) in the upper bar of the notebook, and try to correct your code. If it is the correct cost, take a break and come back in 5 minutes!

# Loading the dataset
print("Loading Training Data...")
X_train, Y_train = ld.load_dataset("TrainData.csv")
print("Loading CrossValidation Data...")
X_cv, Y_cv = ld.load_dataset("CrossValidationData.csv")
print("Data loading compeleted.")
print("number of training examples = " + str(X_train.shape[1]))
print("number of cv examples = " + str(X_cv.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_cv shape: " + str(X_cv.shape))
print("Y_cv shape: " + str(Y_cv.shape))

# Train
print("Start Training...")
parameters = nn.model(X_train, Y_train, X_cv, Y_cv)
W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']        
np.savetxt("parameters/W1.txt",W1)
np.savetxt("parameters/b1.txt",b1)
np.savetxt("parameters/W2.txt",W2)
np.savetxt("parameters/b2.txt",b2)
np.savetxt("parameters/W3.txt",W3)
np.savetxt("parameters/b3.txt",b3)
print("Parameters have been saved!")
