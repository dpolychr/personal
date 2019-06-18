insurance_df.drop('favorite color', axis=1) # axis = 1 means we're dropping a column instead of a row

# Seaborn's pairplot is excellent to visually explore small to medium sized datasets. One by one comparison of each numeric feature in the dataset in the form of a scatterplot plus, diagonally, a view of the distribution of each feature

# Visually detecting redundant features
# Data visualization is a crucial step in any data exploration. Let's use Seaborn to explore some samples of the US Army ANSUR body measurement dataset.
#
# Two data samples have been pre-loaded as ansur_df_1 and ansur_df_2.
#
# Seaborn has been imported as sns.

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Correct! Extracted features can be quite hard to interpret.

# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# Powerful technique to visualize high dimensional data using feature extraction

# Before we apply t-SNE we are going to remove all non-numeric columns by passing a list with the unwanted column names to the pandas df
non_numeric = ['BMI_class', 'Height_class', 'Gender', 'Component']

df_numeric = df_drop(non_numeric, axis = 1)

df_numeric.shape

# t-SNE does not work with non-numeric data as such

# Fitting t-SNE to the ANSUR data
# t-SNE is a great technique for visual exploration of high dimensional datasets. In this exercise, you'll apply it to the ANSUR dataset. You'll remove non-numeric columns from the pre-loaded dataset df and fit TSNE to his numeric dataset.

# Drop the non-numeric columns from the dataset.
# Create a TSNE model with learning rate 50.
# Fit and transform the model on the numeric dataset.

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features)

# t-SNE visualisation of dimensionality
# Time to look at the results of your hard work. In this exercise, you will visualize the output of t-SNE dimensionality reduction on the combined male and female Ansur dataset. You'll create 3 scatterplots of the 2 t-SNE features ('x' and 'y') which were added to the dataset df. In each scatterplot you'll color the points according to a different categorical variable.
#
# seaborn has already been imported as sns and matplotlib.pyplot as plt.

# Use seaborn's sns.scatterplot to create the plot.
# Color the points by 'Component'.

# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Component', data=df)

# Show the plot
plt.show()

# Color the points of the scatterplot by 'Branch'.

# Color the points by Army Branch
sns.scatterplot(x="x", y="y", hue='Branch', data=df)

# Show the plot
plt.show()

# Color the points of the scatterplot by 'Gender'.

# Color the points by Gender
sns.scatterplot(x="x", y="y", hue='Gender', data=df)

# Show the plot
plt.show()

# Models tend to overfit badly on high-dimensional data. So reduce dimensionality. But which features should you drop?

# Split the data into a 70% train and 30% test set using scikit le

from sklearn.metrics import accuracy_score
print(accuracy_score((y_test, svc.predict(X_test))))

# If we want to know whether our model overfitted to the dataset we can have a look at the accuracy on the training set. If this accuracy is much higher than that on the test set we can conclude that the model did not generalize well but simply memorized all training examples
# If we want to improve the accuracy of the model we ll have to add features to the dataset

# To avoid overfitting, the nr of observations should increase exponentially with the number of features. Since this becomes really problematic for high dimensional datasets, this phenomenon is known as the curse of dimensionality. Thus we apply dimensionality reduction

# Train - test split
# In this chapter, you will keep working with the ANSUR dataset. Before you can build a model on your dataset, you should first decide on which feature you want to predict. In this case, you're trying to predict gender.
#
# You need to extract the column holding this feature from the dataset and then split the data into a training and test set. The training set will be used to train the model and the test set will be used to check its performance on unseen data.
#
# ansur_df has been pre-loaded for you.

# Import the train_test_split function from sklearn.model_selection.
# Assign the 'Gender' column to y.
# Remove the 'Gender' column from the dataframe and assign the result to X.
# Set the test size to 30% to perform a 70% train and 30% test data split.

# Import train_test_split()
from sklearn.model_selection import train_test_split

# Select the Gender column as the feature to be predicted (y)
y = ansur_df['Gender']

# Remove the Gender column to create the training data
X = ansur_df.drop('Gender', axis=1)

# Perform a 70% train and 30% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("{} rows in test set vs. {} in training set. {} Features.".format(X_test.shape[0], X_train.shape[0], X_test.shape[1]))

# Fitting and testing the model
# In the previous exercise, you split the dataset into X_train, X_test, y_train, and y_test. These datasets have been pre-loaded for you. You'll now create a support vector machine classifier model (SVC()) and fit that to the training data. You'll then calculate the accuracy on both the test and training set to detect overfitting.

# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics
# Create an instance of the Support Vector Classification class (SVC()).
# Fit the model to the training data.
# Calculate accuracy scores on both train and test data.

# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the training data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))

# 49.7% accuracy on test set vs. 100.0% on training set

# Well done! Looks like the model badly overfits on the training data. On unseen data it performs worse than a random selector would.

# Accuracy after dimensionality reduction
# You'll reduce the overfit with the help of dimensionality reduction. In this case, you'll apply a rather drastic form of dimensionality reduction by only selecting a single column that has some good information to distinguish between genders. You'll repeat the train-test split, model fit and prediction steps to compare the accuracy on test vs. training data.
#
# All relevant packages and y have been pre-loaded.

# Select just the neck circumference ('neckcircumferencebase') column from ansur_df.
# Split the data, instantiate a classifier and fit the data. This has been done for you.
# Once again calculate the accuracy scores on both training and test set.

# Assign just the 'neckcircumferencebase' column from ansur_df to X
X = ansur_df[['neckcircumferencebase']]

# Split the data, instantiate a classifier and fit the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))

# 93.3% accuracy on test set vs. 94.9% on training set

# Wow, what just happened!? On the full dataset the model is rubbish but with a single feature we can make good predictions? This is an example of the curse of dimensionality! The model badly overfits when we feed it too many features. It overlooks that neck circumference by itself is pretty different for males and females.

# Features with missing values or little variance

# Low variance features are so similar between different observation that they may contain little information we can use in an analysis
# To remove them we can use
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=1) # set the minimal variance threshold

# Fit the selector to our dataset
sel.fit(ansur_df)

mask = sel.get_support() # This will give us a TRUE or FALSE value on whether each feature's variance is above the threshold or not

# loc method and specify we want to select all rows using a colon for the first argument and subselect the columns by using our mask as the second
reduced_df = ansur_df.loc[:, mask]

# Normalize the variance before using it for feature selection. To do so we divide each column by its mean value before fitting the selector
sel = VarianceThreshold(threshold=0.005)

sel.fit(ansur_df / ansur_df.mean())

# After normlisation the variance in the dataset will be lower.

# When we apply the selector to our dataset the nr of features is more than haved, to 45

# identifying missing values
pokemon_df.isna()

pokemon_df.isna().sum()  # Total nr of missing values in each column

# Ratio of missing values between zero and one
pokemon_df.isna().sum()  / len(pokemon_df)

# Based on this ration we can create a mask for features that have fewer missing values than a certain threshold
mask = pokemon_df.isna().sum() / len(pokemon_df) < 0.3
print(mask)

# Pass our mask to the loc[] method to subselect the cols
reduced_df = pokemon_df.loc[:, mask]

# Features with low variance
# In the previous exercise you established that 0.001 is a good threshold to filter out low variance features in head_df after normalization. Now use the VarianceThreshold feature selector to remove these features.

# Create the variance threshold selector with a threshold of 0.001.
# Normalize the head_df dataframe by dividing it by its mean values and fit the selector.
# Create a boolean mask from the selector using .get_support().
# Create a reduced dataframe by passing the mask to the .loc[] method.

from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:, mask]

print("Dimensionality reduced from {} to {}.".format(head_df.shape[1], reduced_df.shape[1]))

# Visualizing the correlation matrix
# Reading the correlation matrix of ansur_df in its raw, numeric format doesn't allow us to get a quick overview. Let's improve this by removing redundant values and visualizing the matrix using seaborn.
#
# Seaborn has been pre-loaded as sns, matplotlib.pyplot as plt, NumPy as np and pandas as pd.

# Create the correlation matrix.
# Visualize it using Seaborn's heatmap function.

# Create the correlation matrix
corr = ansur_df.corr()

# Draw the heatmap
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# When we pass this mask to the pandas df mask method it will replace all positions in the df where the mask has a true value with NA

# Use list comprehension to find all cols that have a correlation to any feature stronger than the threshold value
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]

# Feature extraction techniques. These remove correlated features for you

# Be cautious of correlation when features are not linear: Anscombe's quartet

# Filtering out highly correlated features
# You're going to automate the removal of highly correlated features in the numeric ANSUR dataset. You'll calculate the correlation matrix and filter out columns that have a correlation coefficient of more than 0.95 or less than -0.95.
#
# Since each correlation coefficient occurs twice in the matrix (correlation of A to B equals correlation of B to A) you'll want to ignore half of the correlation matrix so that only one of the two correlated features is removed. Use a mask trick for this purpose.

# Calculate the correlation matrix of ansur_df and take the absolute value of this matrix.
# Create a boolean mask with True values in the upper right triangle and apply it to the correlation matrix.
# Set the correlation coefficient threshold to 0.95.
# Drop all the columns listed in to_drop from the dataframe.

# Calculate the correlation matrix and take the absolute value
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))

# Nuclear energy and pool drownings
# The dataset that has been pre-loaded for you as weird_df contains actual data provided by the US Centers for Disease Control & Prevention and Deptartment of Energy.
#
# Let's see if we can find a pattern.
#
# Seaborn has been pre-loaded as sns and matplotlib.pyplot as plt.

# Create a scatterplot with nuclear energy production on the x-axis and the number of pool drownings on the y-axis.

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x="nuclear_energy", y="pool_drownings", data=weird_df)
plt.show()

# Print out the correlation matrix of weird_df
print(weird_df.corr())

# Inspecting the feature coefficients
print(lr.coef_)

# Some values are pretty close to 0. Features with coefficients close to 0 will contribute little to the end result

# zip function to transform the output into a dictionary that shows which feature has which coefficient
print(dict(zip(X.columns, abs(lr.coef_[0]))))

# The fact that we standardized the data first makes sure that we can compare the coefficients to one another

# Recursive Feature Elimination (RFE) is a feature selection algorithm that can be wrapped around any model that produces feature coefficients or feature importance values
# We can pass it the model we want to use and the nr of features we want to select

from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=1)

# Once RFE is done we can check the support_ attribute that contains True/False values to see which features were kept in the dataset
X.columns[rfe.support_]

# Building a diabetes classifier
# You'll be using the Pima Indians diabetes dataset to predict whether a person has diabetes using logistic regression. There are 8 features and one target in this dataset. The data has been split into a training and test set and pre-loaded for you as X_train, y_train, X_test, and y_test.
#
# A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr.

# Fit the scaler on the training features and transform these features in one go.
# Fit the logistic regression model on the scaled training data.
# Scale the test features.
# Predict diabetes presence on the scaled test set.

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred)))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Manual Recursive Feature Elimination
# Now that we've created a diabetes classifier, let's see if we can reduce the number of features without hurting the model accuracy too much.
#
# On the second line of code the features are selected from the original dataframe. Adjust this selection.
#
# A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr.
#
# All necessary functions and packages have been pre-loaded too.

# First, run the given code, then remove the feature with the lowest model coefficient from X.

# Remove the feature with the lowest model coefficient
X = diabetes_df[['glucose', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Automatic Recursive Feature Elimination
# Now let's automate this recursive process. Wrap a Recursive Feature Eliminator (RFE) around our logistic regression estimator and pass it the desired number of features.
#
# All the necessary functions and packages have been pre-loaded and the features have been scaled for you.

# Create the RFE with a LogisticRegression() estimator and 3 features to select.
# Print the features and their ranking.
# Print the features that are not eliminated.

# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc))

# Some models will perform feature selection by design to avoid overfitting. One of those is the RF classifier
# Ensemble model that will pass different, random, subsets of features to a number of decision trees

# The RF algorithm manages to calculate feature importance values

# These values can be extracted from a trained model using the feature_importances attribute
# Just like the coefficients produced by the logistic regressor, these feature importance values can be used to perform feature selection since for unimportant features they will be close to 0.

# One advantage of these feature importance values over coefficients is that they are comparable between features by default, since they always sum up to 1
# Which means we do not have to scale our input data first

# RFE (Recursive Feature Eliminator) with RF
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=6, verbose=1)
rfe.fit(X_train, y_train)

# step parameter to RFE. Here we've set it to 10 so that on each iteration the 10 least important features are dropped.

# Building a random forest model
# You'll again work on the Pima Indians dataset to predict whether an individual has diabetes. This time using a random forest classifier. You'll fit the model on the training data after performing the train-test split and consult the feature importance values.
#
# The feature and target datasets have been pre-loaded for you as X and y. Same goes for the necessary packages and functions.

# Set a 25% test size to perform a 75%-25% train-test split.
# Fit the random forest classifier to the training data.
# Calculate the accuracy on the test set.
# Print the feature importances per feature.

# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))

# Good job! The random forest model gets 78% accuracy on the test set and 'glucose' is the most important feature (0.21

# Random forest for feature selection
# Now lets use the fitted random model to select the most important features from our input dataset X.
#
# The trained model from the previous exercise has been pre-loaded for you as rf.

# Create a mask for features with an importance higher than 0.15.

# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Prints out the mask
print(mask)

# Sub-select the most important features by applying the mask to X.

# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)

# Recursive Feature Elimination with random forests
# You'll wrap a Recursive Feature Eliminator around a random forest model to remove features step by step. This method is more conservative compared to selecting features after applying a single importance threshold. Since dropping one feature can influence the relative importances of the others.
#
# You'll need these pre-loaded datasets: X, X_train, y_train.
#
# Functions and classes that have been pre-loaded for you are: RandomForestClassifier(), RFE(), train_test_split()

# Create a recursive feature eliminator that will select the 2 most important features using a random forest model.

# Wrap the feature eliminator around the random forest model
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)

# Fit the recursive feature eliminator to the training data.

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask using an attribute of rfe
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)

# Change the settings of RFE() to eliminate 2 features at each step

# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)

# How to reduce dimensionality using classification algorithms. Let's see what we can do with regressions

# 20 is the intercept; 5, 2 and 0 are the coefficients of our features, they determine how big an effect each has on the target
# The third feature has a coefficient of 0 and will therefore have no effect on the target whatsoever
# It would be best to remove it from the dataset
y = 20 + 5x1 + 2x2 + 0x3 + error

# LinearRegression has a coeff attribute
lr.coef_

# Contains a NumPy array with the number of elements equal to the number of input features

# To check how accurate the model's predictions are we can calculate the R-squared value on the test set
print(lr.score(X_test, y_test))

# Model will try to find optimal values for the intercept and coefficients by minimizing a loss function
# This function contains the mean sum of the squared differences between actual and predicted values

# Minimizing the MSE (mean squared error) makes the model as accurate as possible.

# However we do not want our model to be super accurate on the training set if that means it no longer generalizes to new data. To avoid this we can introduce regularization

# Tries to make model simple by keeping coefficients low. The strength of regularization can be tweeked with a

# a too low: model might overfit
# when it's too high model might become too simple

# Creating a LASSO regressor
# You'll be working on the numeric ANSUR body measurements dataset to predict a persons Body Mass Index (BMI) using the pre-imported Lasso() regressor. BMI is a metric derived from body height and weight but those two features have been removed from the dataset to give the model a challenge.
#
# You'll standardize the data first using the StandardScaler() that has been instantiated for you as scaler to make sure all coefficients face a comparable regularizing force trying to bring them down.
#
# All necessary functions and classes plus the input datasets X and y have been pre-loaded.

# Set the test size to 30% to get a 70-30% train test split.
# Fit the scaler on the training features and transform these in one go.
# Create the Lasso model.
# Fit it to the scaled training data.

# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Create the Lasso model
la = Lasso()

# Fit it to the standardized training data
la.fit(X_train_std, y_train)

# Lasso model results
# Now that you've trained the Lasso model, you'll score its predictive capacity (R2) on the test set and count how many features are ignored because their coefficient is reduced to zero.
#
# The X_test and y_test datasets have been pre-loaded for you.
#
# The Lasso() model and StandardScaler() have been instantiated as la and scaler respectively and both were fitted to the training data.

# Transform the test set with the pre-fitted scaler.
# Calculate the R2 value on the scaled test data.
# Create a list that has True values when coefficients equal 0.
# Calculate the total number of features with a coefficient of 0.

# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on X_test_std
r_squared = la.score(X_test_std, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))

# Good! We can predict almost 85% of the variance in the BMI value using just 9 out of 91 of the features. The R^2 could be higher though.

# Adjusting the regularization strength
# Your current Lasso model has an R2 score of 84.7%. When a model applies overly powerful regularization it can suffer from high bias, hurting its predictive power.
#
# Let's improve the balance between predictive power and model simplicity by tweaking the alpha parameter.

# Find the highest value for alpha that keeps the R2 value above 98% from the options: 1, 0.5, 0.1, and 0.01

# Find the highest alpha value with R-squared above 98%
la = Lasso(alpha = 0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train_std, y_train)
r_squared = la.score(X_test_std, y_test)
n_ignored_features = sum(la.coef_ == 0)

# Print peformance stats
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))
print("{} out of {} features were ignored.".format(n_ignored_features, len(la.coef_)))

# The LassoCV() class will use cross validation to try out different alpha settings and select the best one

# To remove the features to which the Lasso regressor assigned a zero coefficient, we once again create a mask with True values for all non-zero coefficients

# Feature selection with LassoCV

# Feature selection with RF

# Like RFs, gradient boosting is an ensemble method that will calculate feature importance values

# Choose at least two models voting for a feature in order to keep it

# Creating a LassoCV regressor
# You'll be predicting biceps circumference on a subsample of the male ANSUR dataset using the LassoCV() regressor that automatically tunes the regularization strength (alpha value) using Cross-Validation.
#
# The standardized training and test data has been pre-loaded for you as X_train, X_test, y_train, and y_test.

# Create and fit the LassoCV model on the training set.
# Calculate R2 on the test set.
# Create a mask for coefficients not equal to zero.

from sklearn.linear_model import LassoCV

# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))

# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print('The model explains {0:.1%} of the test set variance'.format(r_squared))

# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))

# Ensemble models for extra votes
# The LassoCV() model selected 27 out of 32 features. Not bad, but not a spectacular dimensionality reduction either. Let's use two more models to select the 10 features they consider most important using the Recursive Feature Eliminator (RFE).
#
# The standardized training and test data has been pre-loaded for you as X_train, X_test, y_train, and y_test.

# Select 10 features with RFE on a GradientBoostingRegressor and drop 3 features on each step.

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R2 on the test set.

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array of the fitted model to gb_mask
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
gb_mask = rfe_gb.support_

# Modify the first step to select 10 features with RFE on a RandomForestRegressor() and drop 3 features on each step.
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
rfe_rf = RFE(estimator=RandomForestRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_rf.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_rf.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
rf_mask = rfe_rf.support_

# Good job! Inluding the Lasso linear model from the previous exercise, we now have the votes from 3 models on which features are important.

# Combining 3 feature selectors
# We'll combine the votes of the 3 models you built in the previous exercises, to decide which features are important into a meta mask. We'll then use this mask to reduce dimensionality and see how a simple linear regressor performs on the reduced dataset.
#
# The per model votes have been pre-loaded as lcv_mask, rf_mask, and gb_mask and the feature and target datasets as X and y.

# Sum the votes of the three models using np.sum().

# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)
print(votes)

# Create a mask for features selected by all 3 models.

# Apply the dimensionality reduction on X and print which features were selected.

# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

# Create a mask for features selected by all 3 models
meta_mask = votes >= 3

# Apply the dimensionality reduction on X
X_reduced = X.loc[:, meta_mask]
print(X_reduced.columns)

# Plug the reduced dataset into the code for simple linear regression that has been written for you.

# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

# Create a mask for features selected by all 3 models
meta_mask = votes >= 3

# Apply the dimensionality reduction on X
X_reduced = X.loc[:, meta_mask]

# Plug the reduced dataset into a linear regression pipeline
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=0)
lm.fit(scaler.fit_transform(X_train), y_train)
r_squared = lm.score(scaler.transform(X_test), y_test)
print('The model can explain {0:.1%} of the variance in the test set using {1:} features.'.format(r_squared, len(lm.coef_)))

# Awesome! Using the votes from 3 models you were able to select just 7 features that allowed a simple linear model to get a high accuracy!

# For PCA it's important to scale the features first so that their values are easier to compare

# The coordinates that each point has in this new reference system are called PC

# Manual feature extraction I
# You want to compare prices for specific products between stores. The features in the pre-loaded dataset sales_df are: storeID, product, quantity and revenue. The quantity and revenue features tell you how many items of a particular product were sold in a store and what the total revenue was. For the purpose of your analysis it's more interesting to know the average price per product.

# Calculate the product price from the quantity sold and total revenue.
# Drop the quantity and revenue features from the dataset.

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['quantity', 'revenue'], axis=1)

print(reduced_df.head())

# Manual feature extraction II
# You're working on a variant of the ANSUR dataset, height_df, where a person's height was measured 3 times. Add a feature with the mean height to the dataset and then drop the 3 original features.

# Add a feature with the mean height to the dataset. Use the .mean() method with axis=1.
# Drop the 3 original height features from the dataset.

# Calculate the mean height
height_df['height'] = height_df[['height_1', 'height_2', 'height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1', 'height_2', 'height_3'], axis=1)

print(reduced_df.head())

# When you are dealing with a dataset with a lot of correlation, the explained variance typically becomes concentrated in the first few components. The remaining components then explain so little variance that they can be dropped

# If you have a lot of highly correlated features, then PCA :-D

# Calculating Principal Components
# You'll visually inspect a 4 feature sample of the ANSUR dataset before and after PCA using Seaborn's pairplot(). This will allow you to inspect the pairwise correlations between the features.
#
# The data has been pre-loaded for you as ansur_df.

# Create a Seaborn pairplot to inspect ansur_df.

# Create a pairplot to inspect ansur_df
sns.pairplot(ansur_df)

plt.show()

# Create the scaler and standardize the data.

from sklearn.preprocessing import StandardScaler

# Create the scaler and standardize the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA() instance and fit and transform the standardized data.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create the scaler and standardize the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)

# This changes the numpy array output back to a dataframe
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(pc_df)
plt.show()

# Good job! Notice how, in contrast to the input features, none of the principal components are correlated to one another.

# PCA on a larger dataset
# You'll now apply PCA on a somewhat larger ANSUR datasample with 13 dimensions, once again pre-loaded as ansur_df. The fitted model will be used in the next exercise. Since we are not using the principal components themselves there is no need to transform the data, instead, it is sufficient to fit pca to the data.

# Create the scaler.
# Standardize the data.
# Create the PCA() instance.
# Fit it to the standardized data.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)

# You'll be inspecting the variance explained by the different principal components of the pca instance you created in the previous exercise.

# Print the explained variance ratio per principal component.

# Inspect the explained variance ratio per component
print(pca.explained_variance_ratio_)

# Print the cumulative sum of the explained variance ratio.

# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())

# When you use PCA for dimensionality reduction, you decide how much of the explained variance you're willing to sacrifice

# two popular dimensionality reduction techniques that can identify non-linear patterns in the original high-dimensional feature space for each disease example: t-Distributed Stochastic Neighbour Embedding (t-SNE) and
# Uniform Manifold Approximation and Projection (UMAP).

# Boruta algorithm for a single (L=1) stochastic iteration to identify the
# most important features that drive gene classification based on the entire OMIM disease annotation
# and compare it against the respective features (where applicable) extracted from the disease-specific
# cases.

# PCA is not the preferred algorithm to reduce the dimensionality of categorical datasets

# Understanding the components
# You'll apply PCA to the numeric features of the Pokemon dataset, poke_df, using a pipeline to combine the feature scaling and PCA in one go. You'll then interpret the meanings of the first two components.
#
# All relevant packages and classes have been pre-loaded for you (Pipeline(), StandardScaler(), PCA()).

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=2))])

# Fit the pipeline to the dataset and extract the component vectors.
# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=2))])

# Fit it to the dataset and extract the component vectors
pipe.fit(poke_df)
vectors = pipe.steps[1][1].components_.round(2)

# Print feature effects
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))

# PCA for feature exploration
# You'll use the PCA pipeline you've built in the previous exercise to visually explore how some categorical features relate to the variance in poke_df. These categorical features (Type & Legendary) can be found in a separate dataframe poke_cat_df.
#
# All relevant packages and classes have been pre-loaded for you (Pipeline(), StandardScaler(), PCA())

# Fit and transform the pipeline to poke_df to extract the principal components.

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('reducer', PCA(n_components=2))])

# Fit the pipeline to poke_df and transform the data
pc = pipe.fit_transform(poke_df)

print(pc)

# PCA in a model pipeline
# We just saw that legendary pokemon tend to have higher stats overall. Let's see if we can add a classifier to our pipeline that detects legendary versus non-legendary pokemon based on the principal components.
#
# The data has been pre-loaded for you and split into training and tests datasets: X_train, X_test, y_train, y_test.
#
# Same goes for all relevant packages and classes(Pipeline(), StandardScaler(), PCA(), RandomForestClassifier()).

# Add a scaler, PCA limited to 2 components, and random forest classifier with random_state=0 to the pipeline.

# Build the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=0))])

# Fit pipeline to the training data
# Build the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Prints the explained variance ratio
print(pipe.steps[1][1].explained_variance_ratio_)

# Score the model accuracy on the test set.

# Build the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Score the accuracy on the test set
accuracy = pipe.score(X_test, y_test)

# Prints the model accuracy
print('{0:.1%} test set accuracy'.format(accuracy))

# Repeat the process with 3 extracted components.
# Build the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=3)),
        ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Score the accuracy on the test set
accuracy = pipe.score(X_test, y_test)

# Prints the explained variance ratio and accuracy
print(pipe.steps[1][1].explained_variance_ratio_)
print('{0:.1%} test set accuracy'.format(accuracy))

# Selecting the proportion of variance to keep
# You'll let PCA determine the number of components to calculate based on an explained variance threshold that you decide.
#
# You'll work on the numeric ANSUR female dataset pre-loaded as ansur_df.
#
# All relevant packages and classes have been pre-loaded too (Pipeline(), StandardScaler(), PCA()).

# Pipe a scaler to PCA selecting 80% of the variance.

# Pipe a scaler to PCA selecting 80% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=0.8))])

# Fit the pipe to the data.
# Pipe a scaler to PCA selecting 80% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=0.8))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print('{} components selected'.format(len(pipe.steps[1][1].components_)))

# Increase the proportion of variance to keep to 90%.
# Let PCA select 90% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=0.9))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print('{} components selected'.format(len(pipe.steps[1][1].components_)))

# Choosing the number of components
# You'll now make a more informed decision on the number of principal components to reduce your data to using the "elbow in the plot" technique. One last time, you'll work on the numeric ANSUR female dataset pre-loaded as ansur_df.
#
# All relevant packages and classes have been pre-loaded for you (Pipeline(), StandardScaler(), PCA()).

# Create a pipeline with a scaler and PCA selecting 10 components.
# Pipeline a scaler and PCA selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=10))])

# Fit the pipe to the data.
# Pipeline a scaler and pca selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=10))])

# Fit the pipe to the data
pipe.fit(ansur_df)

# Plot the explained variance ratio.
# Pipeline a scaler and pca selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=10))])

# Fit the pipe to the data
pipe.fit(ansur_df)

# Plot the explained variance ratio
plt.plot(pipe.steps[1][1].explained_variance_ratio_)

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()

# PCA for image compression
# You'll reduce the size of 16 images with hand written digits (MNIST dataset) using PCA.
#
# The samples are 28 by 28 pixel gray scale images that have been flattened to arrays with 784 elements each (28 x 28 = 784) and added to the 2D numpy array X_test. Each of the 784 pixels has a value between 0 and 255 and can be regarded as a feature.
#
# A pipeline with a scaler and PCA model to select 78 components has been pre-loaded for you as pipe. This pipeline has already been fitted to the entire MNIST dataset except for the 16 samples in X_test.
#
# Finally, a function plot_digits has been created for you that will plot 16 images in a grid.

# Plot the MNIST sample data.
