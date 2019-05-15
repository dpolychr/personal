# Getting to know your data
# Pandas is one the most popular packages used to work with tabular data in Python. It is generally imported using the alias pd and can be used to load a CSV (or other delimited files) using read_csv().
#
# You will be working with a modified subset of the Stackoverflow survey response data in the first three chapters of this course. This data set records the details, and preferences of thousands of users of the StackOverflow website.

# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)

# Selecting specific data types
# Often a data set will contain columns with several different data types (like the one you are working with). The majority of machine learning models require you to have a consistent data type across features. Similarly, most feature engineering techniques are applicable to only one type of data at a time. For these reasons among others, you will often want to be able to access just the columns of certain types when working with a DataFrame.
#
# The DataFrame (so_survey_df) from the previous exercise is available in your workspace.

# Create a subset of so_survey_df consisting of only the numeric (int and float) columns.
# Print the column names contained in so_survey_df_num.

# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)

# One-hot encoding: Converts N categories to N features; Expainable features
# Dummy encoding: Creates N-1 features for N categories omitting the first category. The base value is encoded by the absence of all other values; Necessary info without duplication

# Limiting your columns
counts = df['Country'].value_counts()
print(counts)

# One-hot encoding and dummy variables
# To use categorical variables in a machine learning model, you first need to represent them in a quantitative way. The two most common approaches are to one-hot encode the variables using or to use dummy variables. In this exercise, you will create both types of encoding, and compare the created column sets. We will continue using the same DataFrame from previous lesson loaded as so_survey_df and focusing on its Country column.

# One-hot encode the Country column, adding "OH" as a prefix for each column.
# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')

# Print the columns names
print(one_hot_encoded.columns)

# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)

# Dealing with uncommon categories
# Some features can have many different categories but a very uneven distribution of their occurrences. Take for example Data Science's favorite languages to code in, some common choices are Python, R, and Julia, but there can be individuals with bespoke choices, like FORTRAN, C etc. In these cases, you may not want to create a feature for each value, but only the more common occurrences.

# Extract the Country column of so_survey_df as a series and assign it to countries.
# Find the counts of each category in the newly created countries series.

# Create a series out of the Country column
countries = so_survey_df.Country

# Get the counts of each category
country_counts = countries.value_counts()

# Print the count values for each category
print(country_counts)

# Create a mask for values occurring less than 10 times in country_counts.
# Print the first 5 rows of the mask.

# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Print the top 5 rows in the mask series
print(mask.head())

# Label values occurring less than the mask cutoff as 'Other'.
# Print the new category counts in countries.

# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())

# Binarizing columns
# While numeric values can often be used without any feature engineering, there will be cases when some form of manipulation can be useful. For example on some occasions, you might not care about the magnitude of a value but only care about its direction, or if it exists at all. In these situations, you will want to binarize a column. In the so_survey_df data, you have a large number of survey respondents that are working voluntarily (without pay). You will create a new column titled Paid_Job indicating whether each person is paid (their salary is greater than zero).

# Create a new column called Paid_Job filled with zeros.
# Replace all the Paid_Job values with a 1 where the corresponding ConvertedSalary is greater than 0.

# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df.ConvertedSalary > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())

# Binning values
# For many continuous values you will care less about the exact value of a numeric column, but instead care about the bucket it falls into. This can be useful when plotting values, or simplifying your machine learning models. It is mostly used on continuous variables where accuracy is not the biggest concern e.g. age, height, wages.
#
# Bins are created using pd.cut(df['column_name'], bins) where bins can be an integer specifying the number of evenly spaced bins, or a list of bin boundaries.

# Bin the ConvertedSalary column values into 5 equal bins, in a new column called equal_binned.

# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins = 5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())

# Bin the ConvertedSalary column using the boundaries in the list bins and label the bins using labels.

# Import numpy
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'],
                                         bins = bins, labels = labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())

# How sparse is my data?
# Most data sets contain missing values, often represented as NaN (Not a Number). If you are working with Pandas you can easily check how many missing values exist in each column.
#
# Let's find out how many of the developers taking the survey chose to enter their age (found in the Age column) and their gender (Gender column).

# Subset the DataFrame to only include the 'Age' and 'Gender' columns.
# Print the number of non-missing values in both columns.

# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.info())

# Finding the missing values
# While having a summary of how much of your data is missing can be useful, often you will need to find the exact locations of these missing values. Using the same subset of the StackOverflow data from the last exercise (sub_df), you will show how a value can be flagged as missing.

# Print the top 10 entries of the DataFrame
print(sub_df.head(10))

# Dealing with missing values
# Complete-case analysis or list-wise deletion

# In listwise deletion, all rows with at least one missing value will be dropped
df.dropna(how = 'any')

# Drop rows with missing values in a specific column
df.dropna(subset = ['colname'])

# Replace missing values in a specific col with a given string
df['VersionControl'].fillna(
    value='None Given', inplace=True
)

# Record where the values are not missing
df['SalaryGiven'] = df['ConvertedSalary'].notnull()

# drop a specific column
df.drop(columns=['ConvertedSalary'])

# Listwise deletion
# The simplest way to deal with missing values in your dataset when they are occurring entirely at random is to remove those rows, also called 'listwise deletion'.
#
# Depending on the use case, you will sometimes want to remove all missing values in your data while other times you may want to only remove a particular column if too many values are missing in that column.

# Replacing missing values with constants
# While removing missing data entirely maybe a correct approach in many situations, this may result in a lot of information being omitted from your models.
#
# You may find categorical columns where the missing value is a valid piece of information in itself, such as someone refusing to answer a question in a survey. In these cases, you can fill all missing values with a new category entirely, for example 'No response given'.

