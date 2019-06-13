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




