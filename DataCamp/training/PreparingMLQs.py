# Simple imputation
# As you saw in the last exercise, deleting data can reduce your dataset by too much. In an interview context, this can lead to biased results of your machine learning model.
#
# A more dynamic way of handling missing values is by imputing them. There are a variety of ways of doing this in python, but in this exercise you will be using the SimpleImputer() function from the sklearn.impute module on loan_data.
#
# You will then use pandas and numpy to convert the imputed dataset into a DataFrame.
#
# Note that 2 steps are now added to the pipeline, Instantiate and Fit:
#

# Subset loan_data's numeric columns and assign them to numeric_cols.

# Instantiate a simple imputation object with a mean imputation strategy.
# Fit and transform the data.

# Import imputer module
from sklearn.impute import SimpleImputer

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Impute with mean
imp_mean = SimpleImputer(strategy='mean')
loans_imp_mean = imp_mean.fit_transform(numeric_cols)

# Import imputer module
from sklearn.impute import SimpleImputer

#Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Impute with mean
imp_mean = SimpleImputer(strategy='mean')
loans_imp_mean = imp_mean.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_mean = pd.DataFrame(loans_imp_mean, columns=numeric_cols.columns)

# Import imputer module
from sklearn.impute import SimpleImputer

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Impute with mean
imp_mean = SimpleImputer(strategy='mean')
loans_imp_mean = imp_mean.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_meanDF = pd.DataFrame(loans_imp_mean, columns=numeric_cols.columns)

# Check for missing values
print(loans_imp_meanDF.info())

# Amazing work! You can cookie cutter the exercise code and just set the strategy keyword to 'median' to impute the median or 'most_frequent' to impute the mode, etc. Let's check out the last exercise in this lesson!

# Iterative imputation
# In the previous exercise, you derived mean imputations for missing values of loan_data. However, in a machine learning interview, you will probably be asked about more dynamic imputation techniques that rely on other features in the dataset.
#
# In this exercise, you'll practice a machine-learning based approach for imputing missing values by imputing missing values as a function of remaining features using IterativeImputer() from sklearn.impute.
#
# Note that this function is considered experimental, so please read the documentation for more information.
#
# You're at the same place in the Pipeline:

# Subset loan_data's numeric columns and assign them to numeric_cols.

# Explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# Now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Instantiate an iterative imputation object with 5 iterations and posterior sampling enabled.
# Fit and transform the data.

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)

# Convert return array object back to DataFrame.

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_iterDF = pd.DataFrame(loans_imp_iter, columns=numeric_cols.columns)

# Print the imputed DataFrame's information.

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_iterDF = pd.DataFrame(loans_imp_iter, columns=numeric_cols.columns)

# Check for missing values
print(loans_imp_iterDF.info())

# Awesome! That wraps up this lesson on finding missing data and what you can do about it. Notice that imputation methods follow the first few steps of the machine learning template that you're already familiar with - Import, Instantiate, Fit - but it uses the fit_transform() function instead. We'll discuss the ML template more throughout the rest of the course!

# Train/test distributions
# In a machine learning interview, you will most certainly work with training data and test data. As discussed earlier, poor model performance can result if the distributions of training and test datasets differ.
#
# In this exercise, you'll use functions from sklearn.model_selection as well as seaborn and matplotlib.pyplot to split loan_data into a training set and a test set, as well as visualize their distributions to spot any discrepancies.
#
# The pipeline now includes Train/Test split:

# Subset loan_data to only the Credit Score and Annual Income features, and the target variable Loan Status in that order.
# Create an 80/20 split of loan_data.
# Create pairplots of trainingSet and testSet setting the hue argument to the target variable.

# Create `loan_data` subset: loan_data_subset
loan_data_subset = loan_data[['Credit Score','Annual Income','Loan Status']]

# Create train and test sets
trainingSet, testSet = train_test_split(loan_data_subset, test_size=0.2, random_state=123)

# Examine pairplots
plt.figure()
sns.pairplot(trainingSet, hue='Loan Status', palette='RdBu')
plt.show()

plt.figure()
sns.pairplot(testSet, hue='Loan Status', palette='RdBu')
plt.show()

# Nicely done! Toggle back and forth between the 2 plot matrices if you didn't notice that trainingSet and testSet have different distributions when conditioned on the Loan Status target variable. You'll find out exactly what to do to avoid that in Chapter 2!

# Log and power transformations
# In the last exercise, you compared the distributions of a training set and test set of loan_data.
#
# This is especially poignant in a machine learning interview because the distribution observed dictates whether or not you need to use techniques which nudge your feature distributions toward a normal distribution so that normality assumptions are not violated.
#
# In this exercise, you will be using the log and power transformation from the scipy.stats module on the Years of Credit History feature of loan_data.
#
# All relevant packages have been imported for you.
#
# Here is where you are in the pipeline:

