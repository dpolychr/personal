# Feature engineering
# Most classifier expect numeric features
# Need to convert string columns to numbers
# Use LabelEncoder from sklearn.preprocessing

# Feature engineering
# You are tasked to predict whether a new cohort of loan applicants are likely to default on their loans. You have a historical dataset and wish to train a classifier on it. You notice that many features are in string format, which is a problem for your classifiers. You hence decide to encode the string columns numerically using LabelEncoder(). The function has been preloaded for you from the preprocessing submodule of sklearn. The dataset credit is also preloaded, as is a list of all column names whose data types are string, stored in non_numeric_columns.

# Inspect the first three lines of your data using .head().
# For each column in non_numeric_columns, replace the string values with numeric values using LabelEncoder().
# Confirm your code worked by printing the data types in the credit data frame.

# Inspect the first few lines of your data using head()
credit.head(n=3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)

# Your first pipeline
# Your colleague has used AdaBoostClassifier for the credit scoring dataset. You want to also try out a random forest classifier. In this exercise, you will fit this classifier to the data and compare it to AdaBoostClassifier. Make sure to use train/test data splitting to avoid overfitting. The data is preloaded and transformed so that all features are numeric. The features are available as X and the labels as y. The module RandomForestClassifier has also been preloaded.

# Split the data into train (X_train and y_train) and test (X_test and y_test). Use 20% of the examples for the test set.

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Fit a RandomForest classifier to the training set.
# Predict the labels of the test data using this classifier.

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Use accuracy_score to assess the performance of your classifier. An empty dictionary called accuracies is pre-loaded in your environment.

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Assess the accuracy of both classifiers
accuracies['rf'] = accuracy_score(y_test, rf_predictions)

# Tune model complexity with GridSearchCV()
param_grid = {'max_depth': [5, 10, 20]}

# in-sample accuracy: accuracy on the same data used for training
# Performance using cross-validation also known as out-of-sample accuracy
# Out-of-sample performance actually drops for depths greater than 10, due to overfitting
# More complex is not always better

# Grid search CV for model complexity
# In the last slide, you saw how most classifiers have one or more hyperparameters that control its complexity. You also learned to tune them using GridSearchCV(). In this exercise, you will perfect this skill. You will experiment with:
#
# The number of trees, n_estimators, in a RandomForestClassifier.
# The maximum depth, max_depth, of the decision trees used in an AdaBoostClassifier.
# The number of nearest neighbors, n_neighbors, in KNeighborsClassifier

