# Defining constants with convenience functions
# A constant is the simplest category of tensor. It can't be trained, which makes it a bad choice for a model's parameters, but a good choice for input data. Input data may be transformed after it is defined or loaded, but is typically not modified by the training process.
#
# In this exercise, we will practice defining constants using some of the operations discussed in the video. Note that we have not imported the entire tensorflow API and will not import it for most exercises. You can complete this exercise using the operations fill(), ones_like(), and constant(), which have been imported from tensorflow version 2.0 and are available in the IPython shell.

# Define a 3x4 tensor, x, using fill() where each element has the value 9.
# Create a tensor, y, with the same shape as x, but with all values set to one using ones_like().
# Define a 1-dimensional constant(), z, which contains the following elements: [1, 2, 3, 4].
# Print z as a numpy array using .numpy().

# Define a 3x4 tensor with all values equal to 9
x = fill([3, 4], 9)

# Define a tensor of ones with the same shape as x
y = ones_like(x)

# Define the one-dimensional vector, z
z = constant([1, 2, 3, 4])

# Print z as a numpy array
print(z.numpy())

# Defining variables
# Unlike a constant, a variable's value can be modified. This will be quite useful when we want to train a model by updating its parameters. Constants can't be used for this purpose, so variables are the natural choice.
#
# Let's try defining and working with a variable. Note that Variable(), which is used to create a variable tensor, has been imported from tensorflow and is available to use in the exercise.

# Define a variable, X, as the 1-dimensional tensor: [1, 2, 3, 4].
# Print X. What did this tell you?
# Apply .numpy() to X and assign it to Z.
# Print Z. What did this tell you?

# Define the 1-dimensional variable X
X = Variable([1, 2, 3, 4])

# Print the variable X
print(X)

# Convert X to a numpy array and assign it to Z
Z = X.numpy()

# Print Z
print(Z)

# Checking properties of tensors
# In later chapters, you will make use of constants and variables to train models. Your datasets will be represented as constant tensors of type tf.Tensor() or numpy arrays. The model's parameters will be represented by variables that are updated during computation.
#
# In this exercise, you will examine the properties of two tensors: X and Y. Note that they have already been defined and are available in the Python shell. Use the print() function to determine which statement about X and Y is true.

# Tensorflow has a model of computation that revolves around the use of graphs

from tensorflow import constant, add

# Define 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])

# Define 1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])

# Define 2-dimensional tensors
A2 = constant([[1,2], [3,4]])
B2 = constant([[5,6], [7,8]])

# Tensor addition
C0 = add(A0, B0) # scalar addition
C1 = add(A1, B1) # vector addition
C2 = add(A2, B2) # matrix addition

# The add() operation performs element-wise addition with two tensors
# Element-wise addition requires both tensors to have the same shape

# Multiplication in TensorFlow
# Element-wise multiplication performed using multiply(). Tensors multiplied must have the same shape

# Matrix multiplication performed using matmul(). Nr of columns of A must equal the nr of rows of B

from tensorflow import ones, reduce_sum

# Define a 2x3x4 tensor of ones
A = ones([2, 3, 4])

# Sum over all dimensions
B = reduce_sum(A)

# Sum over dimensions 0, 1 and 2
B0 = reduce_sum(A, 0) # 3x4 matrix of 2s
B1 = reduce_sum(A, 1) # 2x4 matrix of 3s
B2 = reduce_sum(A, 2) # 2x3 matrix of 4s

# Performing element-wise multiplication
# Element-wise multiplication in TensorFlow is performed using two tensors with identical shapes. This is because the operation multiplies elements in corresponding positions in the two tensors. An example of an element-wise multiplication, denoted by the ⊙ symbol, is shown below:
#
# [1221]⊙[3215]=[3425]
# In this exercise, you will perform element-wise multiplication, paying careful attention to the shape of the tensors you multiply. Note that multiply(), constant(), and ones_like() have been imported for you.

# Define the tensors A0 and B0 as constants.
# Set A1 to be a tensor of ones with the same shape as A0.
# Set B1 to be a tensor of ones with the same shape as B0.
# Set A2 and B2 equal to the element-wise products of A0 and A1, and B0 and B1, respectively.

# Define tensors A0 and B0 as constants
A0 = constant([1, 2, 3, 4])
B0 = constant([[1, 2, 3], [1, 6, 4]])

# Define A1 and B1 to have the correct shape
A1 = ones_like(A0)
B1 = ones_like(B0)

# Perform element-wise multiplication
A2 = multiply(A0, A1)
B2 = multiply(B0, B1)

# Print the tensors A2 and B2
print(A2.numpy())
print(B2.numpy())

# Making predictions with matrix multiplication
# In later chapters, you will learn to train linear regression models. This process will yield a vector of parameters that can be multiplied by the input data to generate a vector of predictions. In the exercise, we will use the following tensors:
#
# X=⎡⎣⎢⎢⎢⎢125621810⎤⎦⎥⎥⎥⎥ b=[12] y=⎡⎣⎢⎢⎢⎢642023⎤⎦⎥⎥⎥⎥
# X is the matrix of input data, b is the parameter vector, and y is the target vector. You will use matmul() to perform matrix multiplication of X by b to generate predictions, ypred, which you will compare with y. Note that we have imported matmul(), constant(), and subtract(), which subtracts the second argument from the first.

# Define X, b, and y as constants.
# Compute the predicted value vector, ypred, by multiplying the input data, X, by the parameters, b. Use matrix multiplication, rather than the element-wise product.
# Use subtract() to compute the error as the deviation of the predicted values, ypred, from the true values or "targets," y.

# Define X, b, and y as constants
X = constant([[1, 2], [2, 1], [5, 8], [6, 10]])
b = constant([[1], [2]])
y = constant([[6], [4], [20], [23]])

# Compute ypred using X and b
ypred = matmul(X, b)

# Compute and print the error as y - ypred
error = subtract(y, ypred)
print(error.numpy())

# Summing over tensor dimensions
# You've been given a matrix, wealth. This contains the value of bond and stock wealth for five individuals. Note that this is given in thousands of dollars.
#
# wealth = [115072460302510]
# The first row corresponds to bonds and the second corresponds to stocks. Each column gives the stock and bond wealth for a single individual. Use wealth, reduce_sum(), and .numpy() to determine which statements are correct about wealth.

# Finding the optimum of a function

# Reshaping tensors
# In many machine learning problems, you will need to reshape your input data. If, for instance, you loaded a 9-pixel, grayscale image of the letter H, it might have the following 2-dimensional representation:
#
# ⎡⎣⎢⎢25525525502550255255255⎤⎦⎥⎥
# Some models are designed to use image data as an input. Many, however, require you to transform the data into a vector. In this exercise, we will use the reshape() operation to practice transforming tensors. Note that ones() and reshape() operations have been imported.

# Reshape the grayscale image into a 256x1 image_vector and a 4x4x4x4 image_tensor.
# Define input image
image = ones([16, 16])

# Reshape image into a vector
image_vector = reshape(image, (256, 1))

# Reshape image into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4))

# Adapt image by adding three color channels and then adjust image_vector and image_tensor accordingly.
# Add three color channels
image = ones([16, 16, 3])

# Reshape image into a vector
image_vector = reshape(image, (768, 1))

# Reshape image into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4, 3))

