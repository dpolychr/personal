import sklearn.datasets
newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
X, y = newsgroups.data, newsgroups.target
X.shape
y.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # specify model hyperparameters like the nr of neighbors
knn.fit(X, y)
y_pred = knn.predict(X)

# Model evaluation
knn.score(X, y) # Not meaningful since we want to see how the model generalises to unseen data

# We need a validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y) # X_train, y_train: training set; X_test, y_test contain the test or validation set which by default contains 25% of the examples

# our model needs to be trained on the training set only
knn.fit(X_train, y_train)

knn.score(X_test, y_test)

# KNN classification
# In this exercise you'll explore a subset of the Large Movie Review Dataset. The variables X_train, X_test, y_train, and y_test are already loaded into the environment. The X variables contain features based on the words in the movie reviews, and the y variables contain labels for whether the review sentiment is positive (+1) or negative (-1).

# Create a KNN model with default hyperparameters.
# Fit the model.
# Print out the prediction for the test example 0.

from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test[0])
print("Prediction for test example 0:", pred)

# Comparing models
# Compare k nearest neighbors classifiers with k=1 and k=5 on the handwritten digits data set, which is already loaded into the variables X_train, y_train, X_test, and y_test. You can set k with the n_neighbors parameter when creating the KNeighborsClassifier object, which is also already imported into the environment.
#
# Which model has a higher test accuracy?

knn_1 = KNeighborsClassifier(n_neighbors=1)

knn_1.fit(X_train, y_train)

knn_1.score(X_test, y_test)

knn_5 = KNeighborsClassifier(n_neighbors=5)

knn_5.fit(X_train, y_train)

knn_5.score(X_test, y_test)

# Applying Logistic Regression and SVM
# Try the wine classification dataset built into scikit-learn
import sklearn.datasets

wine = sklearn.datasets.load_wine()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(wine.data, wine.target)

lr.score(wine.data, wine.target)

# Using Linear SVC
# in scikit-learn, the basic SVM classifier is called Linear SVC (Support Vector Classifier)
# nonlinear SVMs: more complex models

# Linear SVC
# SVC is the non-linear version # Achieves 100% training accuracy probably due to classifier overfitting

# Hyperparameter is a choice about the model you make before fitting to the data, and often controls the complexity of the model
# If the model is too simple, it may be unable to capture the patterns in the data leading to low training accuracy. This is underfitting
# Overfitting: model is too complex, low test accuracy

# Running LogisticRegression and SVC
# In this exercise, you'll apply logistic regression and a support vector machine to classify images of handwritten digits.

# Apply logistic regression and SVM (using SVC()) to the handwritten digits data set using the provided train/validation split.
# For each classifier, print out the training and validation accuracy.

from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))

# Sentiment analysis for movie reviews
# In this exercise you'll explore the probabilities outputted by logistic regression on a subset of the Large Movie Review Dataset.
#
# The variables X and y are already loaded into the environment. X contains features based on the number of times words appear in the movie reviews, and y contains labels for whether the review sentiment is positive (+1) or negative (-1).

# Train a logistic regression model on the movie review data.
# Predict the probabilities of negative vs. positive for the two given reviews.
# Feel free to write your own reviews and get probabilities for those too!

# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])

# Dividing line between the two regions is called decision boundary. Decision boundary considered linear because it looks like a line

# In regression we are trying to predict a continuous value

# linear classifier: a classifier that learns linear decision boundaries e.g logistic regression, linear SVM

# A dataset is called linearly separable if it can be perfectly explained by a linear classifier

# Visualizing decision boundaries
# In this exercise, you'll visualize the decision boundaries of various classifier types.
#
# A subset of scikit-learn's built-in wine dataset is already loaded into X, along with binary labels in y.

# Create the following classifier objects with default hyperparameters: LogisticRegression, LinearSVC, SVC, KNeighborsClassifier.
# Fit each of the classifiers on the provided data using a for loop.
# Call the plot_4_classifers() function (similar to the code here), passing in X, y, and a list containing the four classifiers.

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(), SVC(), KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X, y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()

# To get the source code of a python function
import inspect
lines = inspect.getsource(foo)
print(lines)


def plot_4_classifiers(X, y, clfs):

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)", "(2)", "(3)", "(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()

# Dot products
import numpy as np
x = np.arange(3)
y = np.arange(3, 6)

x*y

# The sum of these numbers is also known as the dot product
np.sum(x*y)

# x@y gives us the same result
# x@y is called the dot product of x,y and is written x * y

# Using dot products we can express how linear classifiers make predictions

# This is the same for logistic regression and linear SVM
# fit is different but predict is the same

# The predict function computes the raw model output:
# raw model output = coefficients * features + intercept

# checks if it's positive or negative

# returns a result based on the names of the classes in your dataset

# Changing the model coefficients
# When you call fit with scikit-learn, the logistic regression coefficients are automatically learned from your dataset. In this exercise you will explore how the decision boundary is represented by the coefficients. To do so, you will change the coefficients manually (instead of with fit), and visualize the resulting classifiers.
#
# A 2D dataset is already loaded into the environment as X and y, along with a linear classifier object model.

# Set the two coefficients and the intercept to various values and observe the resulting decision boundaries.
# Try to build up a sense of how the coefficients relate to the decision boundary.
# Set the coefficients and intercept such that the model makes no errors on the given training data.

# Set the coefficients
model.coef_ = np.array([[0,1]])
model.intercept_ = np.array([0])

# Plot the data and decision boundary
plot_classifier(X,y,model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)

# Many machine learning algorithms involve minimizing a loss

# Least squares: the squared loss

# Loss is used to fit the model on the data

# Classification errors: the 0-1 loss

# Minimise a loss function using a Python package called scipy.optimize.minimize which can minimize all sorts of actions

# Minimise the square error from linear regression

# Minimizing a loss function
# In this exercise you'll implement linear regression "from scratch" using scipy.optimize.minimize.
#
# We'll train a model on the Boston housing price data set, which is already loaded into the variables X and y. For simplicity, we won't include an intercept in our regression model.

# Fill in the loss function for least squares linear regression.
# Print out the coefficients from fitting sklearn's LinearRegression.

# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)

# Implementing logistic regression
# This is very similar to the earlier exercise where you implemented linear regression "from scratch" using scipy.optimize.minimize. However, this time we'll minimize the logistic loss and compare with scikit-learn's LogisticRegression (we've set C to a large value to disable regularization; more on this in Chapter 3!).
#
# The log_loss() function from the previous exercise is already defined in your environment, and the sklearn breast cancer prediction dataset (first 10 features, standardized) is loaded into the variables X and y.

# Input the number of training examples into range().
# Fill in the loss function for logistic regression.
# Compare the coefficients to sklearn's LogisticRegression.

# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(len(X)):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)

# Regularization makes the coefficients smaller
# more regularization: lower training accuracy
# more regularization: (almost always) higher test accuracy

# For linear regression we are using the terms Ridge and Lasso for two different types of regularization

# Lasso = linear regression with L1 regularization
# Ridge = linear regression with L2 regularization

# Both help reduce overfitting and L1 also performs feature selection
# Scaling features is a good practice, especially when using regularization

# L1 regularization sets many of the coefficients to 0 thus ignoring those features. In other words it performed feature selection for us
# L2 regularization just shrinks the coefficients to be smaller

# Regularized logistic regression
# In Chapter 1 you used logistic regression on the handwritten digits data set. Here, we'll explore the effect of L2 regularization.
#
# The handwritten digits dataset is already loaded, split, and stored in the variables X_train, y_train, X_valid, and y_valid. The variables train_errs and valid_errs are already initialized as empty lists.

# Loop over the different values of C_value, creating and fitting a LogisticRegression model each time.
# Save the error on the training set and the validation set for each model.
# Create a plot of the training and testing error as a function of the regularization parameter, C.
# Looking at the plot, what's the best value of C?

# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)

    # Evaluate error rates and append to lists
    train_errs.append(1.0 - lr.score(X_train, y_train))
    valid_errs.append(1.0 - lr.score(X_valid, y_valid))

# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()

# Logistic regression and feature selection
# In this exercise we'll perform feature selection on the movie review sentiment data set using L1 regularization. The features and targets are already loaded for you in X_train and y_train.
#
# We'll search for the best value of C using scikit-learn's GridSearchCV(), which was covered in the prerequisite course.

# Instantiate a logistic regression object that uses L1 regularization.
# Find the value of C that minimizes cross-validation error.
# Print out the number of selected features for this value of C.

# Specify L1 regularization
lr = LogisticRegression(penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))

# Identifying the most positive and negative words
# In this exercise we'll try to interpret the coefficients of a logistic regression fit on the movie review sentiment dataset. The model object is already instantiated and fit for you in the variable lr.
#
# In addition, the words corresponding to the different features are loaded into the variable vocab. For example, since vocab[100] is "think", that means feature 100 corresponds to the number of times the word "think" appeared in that movie review.

# Find the words corresponding to the 5 largest coefficients.
# Find the words corresponding to the 5 smallest coefficients.

# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten())
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print(vocab[inds_descending[i]], end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print(vocab[inds_ascending[i]], end=", ")
print("\n")

# Logistic regression and probabilities
# So far, logreg to make hard predictions, meaning we predict either one class or another
# Interpret the raw model output as a probability

# C very large so no regularization
# The ratio of the coefficients gives us the slope of the line and the magnitude of the coefficients gives us the confidence levels
# Logistic regression probabilities: "squashed" raw model output. The sigmoid fun takes care of that

# You got it! As you probably noticed, smaller values of C lead to less confident predictions. That's because smaller C means more regularization, which in turn means smaller coefficients, which means raw model outputs closer to zero and, thus, probabilities closer to 0.5 after the raw model output is squashed through the sigmoid function. That's quite a chain of events!

# Visualizing easy and difficult examples
# In this exercise, you'll visualize the examples that the logistic regression model is most, and least, confident about by looking at the largest, and smallest, predicted probabilities.
#
# The handwritten digits dataset is already loaded into the variables X and y. The show_digit function takes in an integer index and plots the corresponding image, with some extra information displayed above the image.

# Fill in the first blank with the index of the digit that the model is most confident about.
# Fill in the second blank with the index of the digit that the model is least confident about.
# Observe the images: do you agree that the first one is more ambiguous than the second?

def show_digit(i, lr=None):
    plt.imshow(np.reshape(X[i], (8,8)), cmap='gray', vmin = 0, vmax = 16, interpolation=None)
    plt.xticks(())
    plt.yticks(())
    if lr is None:
        plt.title("class label = %d" % y[i])
    else:
        pred = lr.predict(X[i][None])
        pred_prob = lr.predict_proba(X[i][None])[0,pred]
        plt.title("label=%d, prediction=%d, proba=%.2f" % (y[i], pred, pred_prob))
    plt.show()

lr = LogisticRegression()
lr.fit(X,y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba,axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)

# Fitting multi-class logistic regression
# In this exercise, you'll fit the two types of multi-class logistic regression, one-vs-rest and softmax/multinomial, on the handwritten digits data set and compare the results. The handwritten digits dataset is already loaded and split into X_train, y_train, X_test, and y_test.

# Fit a one-vs-rest logistic regression classifier and report the results.
# Fit a softmax logistic regression classifier by setting the multiclass paramater, plus setting to be solver = "lbfgs", and report the results.

# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression()
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class="multinomial", solver = "lbfgs")
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))

# Visualizing multi-class logistic regression
# In this exercise we'll continue with the two types of multi-class logistic regression, but on a toy 2D data set specifically designed to break the one-vs-rest scheme.
#
# The data set is loaded into X_train and y_train. The two logistic regression objects,lr_mn and lr_ovr, are already instantiated (with C=100), fit, and plotted.
#
# Notice that lr_ovr never predicts the dark blue class... yikes! Let's explore why this happens by plotting one of the binary classifiers that it's using behind the scenes.

# Create a new logistic regression object (also with C=100) to be used for binary classification.
# Visualize this binary classifier with plot_classifier... does it look reasonable?

# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train==1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train==1, lr_class_1)

# Nice work! As you can see, the binary classifier incorrectly labels almost all points in class 1 (shown as red triangles in the final plot)! Thus, this classifier is not a very effective component of the one-vs-rest classifier. In general, though, one-vs-rest often works well.

# One-vs-rest SVM
# As motivation for the next and final chapter on support vector machines, we'll repeat the previous exercise with a non-linear SVM. Once again, the data is loaded into X_train, y_train, X_test, and y_test .
#
# Instead of using LinearSVC, we'll now use scikit-learn's SVC object, which is a non-linear "kernel" SVM (much more on what this means in Chapter 4!). Again, your task is to create a plot of the binary classifier for class 1 vs. rest.

# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train == 1)
plot_classifier(X_train, y_train ==1, svm_class_1)

# Cool, eh?! The non-linear SVM works fine with one-vs-rest on this dataset because it learns to "surround" class 1.

def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None): # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('probability of red $\Delta$ class', fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y==labels[0]], X1[y==labels[0]], cmap=plt.cm.coolwarm, s=60, c='b', marker='o', edgecolors='k')
        ax.scatter(X0[y==labels[1]], X1[y==labels[1]], cmap=plt.cm.coolwarm, s=60, c='r', marker='^', edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel(data.feature_names[0])
#     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
#     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax

# SVMs are linear classifiers (so far)
# Trained using the hinge loss and L2 regularization

# Effect of removing examples
# Support vectors are defined as training examples that influence the decision boundary. In this exercise, you'll observe this behavior by removing non support vectors from the training set.
#
# The wine quality dataset is already loaded into X and y (first two features only). (Note: we specify lims in plot_classifier() so that the two plots are forced to use the same axis limits and can be compared directly.)

# Train a linear SVM on the whole data set.
# Create a new data set containing only the support vectors.
# Train a new linear SVM on the smaller data set.

# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))

# How to fit nonlinear boundaries using linear classifiers
# Not linearly separable means that there is no linear boundary that perfectly classifies all the points
# Fitting a linear model in a transformed space (squared for instance) corresponds to fitting a nonlinear model in the original space!

# Kernels and kernel SVMs implement feature transformation in a computationally efficient way

from sklearn.svm import SVC
svm = SVC(gamma=1) # default is kernel rbf. Many non-linear kernels exist, in this case we'll focus on RBF

# COntrol the shape of the boundary using hyperparameters

# gamma controls the smoothness of the boundary. By decreasing gamma we make the boundaries smoother
# With the right hyperparameters, RBF SVMs are capable of separating almost any dataset!

# GridSearchCV warm-up
# In the video we saw that increasing the RBF kernel hyperparameter gamma increases training accuracy. In this exercise we'll search for the gamma that maximizes cross-validation accuracy using scikit-learn's GridSearchCV. A binary version of the handwritten digits dataset, in which you're just trying to predict whether or not an image is a "2", is already loaded into the variables X and y.

# Create a GridSearchCV object.
# Call the fit() method to select the best value of gamma based on cross-validation accuracy.

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X,y)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Great job! Larger values of gamma are better for training accuracy, but cross-validation helped us find something different (and better!).

# Jointly tuning gamma and C with GridSearchCV
# In the previous exercise the best value of gamma was 0.001 using the default value of C, which is 1. In this exercise you'll search for the best combination of C and gamma using GridSearchCV.
#
# As in the previous exercise, the 2-vs-not-2 digits dataset is already loaded, but this time it's split into the variables X_train, y_train, X_test, and y_test. Even though cross-validation already splits the training set into parts, it's often a good idea to hold out a separate test set to make sure the cross-validation results are sensible.

# Run GridSearchCV to find the best hyperparameters using the training set.
# Print the best values of the parameters.
# Print out the accuracy on the test set, which was not used during the cross-validation procedure.

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

# You got it! Note that the best value of gamma, 0.0001, is different from the value of 0.001 that we got in the previous exercise, when we fixed C=1. Hyperparameters can affect each other!

# SGDClassifier scales well to large datasets. Only have to set the loss hyperparameters in order to switch between logreg and SVM
from sklearn.linear_model import SGDClassifier
logreg = SGDClassifier(loss='log')
linsvm = SGDClassifier(loss='hinge')

# SGDClassifier hyperparameter is alpha is like 1/C, bigger alpha means more regularization

# Using SGDClassifier
# In this final coding exercise, you'll do a hyperparameter search over the regularization type, regularization strength, and the loss (logistic regression vs. linear SVM) using SGDClassifier().

# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'loss':['hinge', 'log'], 'penalty':['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

# Congrats, you finished the last exercise in the course! One advantage of SGDClassifier is that it's very fast - this would have taken a lot longer with LogisticRegression or LinearSVC.

# The prediction can be of two types: either classification in which a class label is assigned to a new data point or regression wherein a value is assigned to the new data point. Unlike classification, in regression, the mean of all the k-nearest neighbors is assigned to the new data point.

# Sklearn is a machine learning python library that is widely used for data-science related tasks. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means, KNN, etc.. Under sklearn you have a library called datasets in which you have multiple datasets that can be used for different tasks including the Iris dataset, all these datasets can be loaded out of the box. It is pretty intuitive and straightforward. So, let's quickly load the iris dataset.

# Loading datasets
from sklearn.datasets import load_iris

import numpy as np

# load_iris has both the data and the class labels for each sample. Let's quickly extract all of it.

data = load_iris().data

# Let's extract the class labels
labels = load_iris().target

labels.shape

labels = np.reshape(labels, (150,1))

# Now, you will use the concatenate function available in the numpy library, and you will use axis=-1 which will concatenate based on the second dimension.

import pandas as pd

data = np.concatenate([data, labels], axis = -1)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']

dataset = pd.DataFrame(data, columns=names)

# Now, you have the dataset data frame that has both data & the class labels that you need!
# Before you dive any further, remember that the labels variable has class labels as numeric values, but you will convert the numeric values as the flower names or species.
#
# For doing this, you will select only the class column and replace each of the three numeric values with the corresponding species. You will use inplace=True which will modify the data frame dataset.

dataset['species'].replace(0, 'Iris-setosa', inplace=True)
dataset['species'].replace(1, 'Iris-versicolor', inplace=True)
dataset['species'].replace(2, 'Iris-virginica',inplace=True)

dataset.head()

# Minimizing a loss function
# In this exercise you'll implement linear regression "from scratch" using scipy.optimize.minimize.
#
# We'll train a model on the Boston housing price data set, which is already loaded into the variables X and y. For simplicity, we won't include an intercept in our regression model.

# Let's visualize the data that you loaded above using a scatterplot to find out how much one variable is affected by the other variable or let's say how much correlation is between the two variables.
#
# You will use matplotlib library to visualize the data using a scatterplot.

import matplotlib.pyplot as plt

plt.figure(4, figsize=(10, 8))

plt.scatter(data[:50, 0], data[:50, 1], c='r', label='Iris-setosa')

plt.scatter(data[50:100, 0], data[50:100, 1], c='g', label='Iris-versicolor')

# Well, the answer is pretty much all the time. It is a good practice to normalize your data as it brings all the samples in the same scale and range. Normalizing the data is crucial when the data you have is not consistent. You can check for inconsistency by using the describe() function that you studied above which will give you max and min values. If the max and min values of one feature are significantly larger than the other feature then normalizing both the features to the same scale is very important.

from sklearn.model_selection import train_test_split

train_data,test_data,train_label,test_label = train_test_split(dataset.iloc[:,:3], dataset.iloc[:,4], test_size=0.2, random_state=42)

train_data.shape,train_label.shape,test_data.shape,test_label.shape

# The kNN model
from sklearn.neighbors import KNeighborsClassifier

# Note: the k (n_neighbors) parameter is often an odd number to avoid ties in the voting scores.

# In order to decide the best value for hyperparameter k, you will do something called grid-search. You will train and test your model on 10 different k values and finally use the one that gives you the best results.

neighbors = np.arange(1, 9)
train_accuracy = np.zeros(len(neighbors))
test_accuracy = np.zeros(len(neighbors))

# Next piece of code is where all the magic will happen. You will enumerate over all the nine neighbor values and for each neighbor you will then predict both on training and testing data. Finally, store the accuracy in the train_accuracy and test_accuracy numpy arrays.

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    knn.fit(train_data, train_label)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(train_data, train_label)

    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(test_data, test_label)

# Next, you will plot the training and testing accuracy using matplotlib, with accuracy vs. varying number of neighbors graph you will be able to choose the best k value at which your model performs the best.

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.title('KNN accuracy with varying number of neighbors',fontsize=20)

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend(prop={'size': 20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# k = 3 is the best
knn = KNeighborsClassifier(n_neighbors=3)

#Fit the model
knn.fit(train_data, train_label)

#Compute accuracy on the training set
train_accuracy = knn.score(train_data, train_label)

#Compute accuracy on the test set
test_accuracy = knn.score(test_data, test_label)

# Evaluating your Model
# In the last segment of this tutorial, you will be evaluating your model on the testing data using a couple of techniques like confusion_matrix and classification_report.
#
# Let's first check the accuracy of the model on the testing data.

# Confusion Matrix
# A confusion matrix is mainly used to describe the performance of your model on the test data for which the true values or labels are known.

prediction = knn.predict(test_data)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=30)
    plt.xlabel('Predicted label',fontsize=30)
    plt.tight_layout()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
class_names = load_iris().target_names

from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=class_names)
plt.title('Confusion Matrix',fontsize=30)
plt.show()

# From the above confusion_matrix plot, you can observe that the model classified all the flowers correctly except one versicolor flower which is classified as a virginica flower.
#
# Classification Report
# Classification report helps you in identifying the misclassified classes in much more detail by giving precision, recall and F1 score for each class. You will use the sklearn library to visualize the classification report.

from sklearn.metrics import classification_report

print(classification_report(test_label, prediction))

