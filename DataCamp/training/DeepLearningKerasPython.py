# Hello nets!
# You're going to build a simple neural network to get a feeling for how quickly it is to accomplish in Keras.
#
# You will build a network that takes two numbers as input, passes them through a hidden layer of 10 neurons, and finally outputs a single non-constrained number.
#
# A non-constrained output can be obtained by avoiding setting an activation function in the output layer. This is useful for problems like regression, when we want our output to be able to take any value.

# Import the Sequential model from keras.models and the Denselayer from keras.layers.
# Create an instance of the Sequential model.
# Add a 10-neuron hidden Dense layer with an input_shape of two neurons.
# Add a final 1-neuron output layer and summarize your model with summary().

# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()

# Counting parameters
# You've just created a neural network. Create a new one now and take some time to think about the weights of each layer. The Keras Dense layer and the Sequential model are already loaded for you to use.
#
# This is the network you will be creating:

# Instantiate a new Sequential() model.
# Add a Dense() layer with five neurons and three neurons as input.
# Add a final dense layer with one neuron and no activation.

# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()

from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape=(2,)))

# Add the ouput layer
model.add(Dense(1))

# Specifying a model
# You will build a simple model to forecast the orbit of the meteor!
#
# Your training data consist of measurements taken at time steps from -10 minutes before the impact region to +10 minutes after. Each time step can be viewed as an X coordinate in our graph, which has an associated position Y for the meteor at that time step.

# This data is stored in two numpy arrays: one called time_steps , containing the features, and another called y_positions, with the labels.
#
# Feel free to look at these arrays in the console anytime, then build your model! Keras Sequential model and Dense layers are available for you to use.

# Instantiate a Sequential model.
# Add a Dense layer of 50 neurons with an input shape of 1 neuron.
# Add two Dense layers of 50 neurons each and 'relu' activation.
# End your model with a Dense layer with a single neuron and no activation.

