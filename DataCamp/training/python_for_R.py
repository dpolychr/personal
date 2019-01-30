# If you want to select every other value from the first 5 elements in the list
# left inclusive right exclusive nature of Python
l = [0, 1, 2, 3, 4]

l[0:5:2]

l[::2]

# You cannot have named lists in python
# Python dics are created with a pair of curly brackets. Each key value pair includes a colon, with the key to the left and the value to the right
# of the colon

# Do not rely on the printed order of the keys
# To extract a value from the dic, you place the key of the dic inside square brackets

# Python is OO. All variables, lists and other data str you create are objects and these objects can have attributes and methods associated with them
# You can't use a period in a variable or a fun name because periods are used in several important contexts

# Add elements to a list using the append method
l.append('appended value')
l

# For dics you can call the update method. You pass update of key-value pairs. If the keys already exist they will be updated with the new value, if not they will be aded to the dictionary

# Methods are special funs that belong to specific objects. For instance, if you try calling append on a dictionary object you will get an error because the method append is not defined for a dictionary but it is defined for a list

# Arrays and dataframes are not built into Python.
# Numpy gives you the array
# In python you denote code blocks with indentations

val = 2

if val == 1:
    print('snap')
elif val == 2:
    print('crackle')
else:
    print('pop')


num_val = [1, 2, 3, 4]

for value in num_val:
    print(value)

num_drinks = [5, 4, 3, 3, 3, 5, 6, 10]

# Write a for loop
for drink in num_drinks:
    # if/else statement
    if drink <= 4:
        print('non-binge')
    else:
        print('binge')

def my_sq(x):
    return x ** 2

# Lambda functions in Python are the same as anonymous funs in R

# R
add_1 <- function(x) x + 1

function(x) x + 1

# Python
def add_1(x):
    return x + 1

all_lam = lambda x: x + 1

# Lambda functions
# If you have ever used the *apply family of functions (such as sapply() and lapply()) in R, there's a good chance you might have used anonymous functions. Anonymous functions in Python are known as lambda functions.
# These functions are not too different from a regular function. The keyword in a lambda function is lambda instead of def. These functions are typically used for functions that are 'one-line' long.

cube_lambda = lambda x: x**3
print(cube_lambda(3))

# A function that takes a value and returns its square
def sq_func(x):
    return (x ** 2)

# A lambda function that takes a value and returns its square
sq_lambda = lambda x: x ** 2

# Use the lambda function
print(sq_lambda(3))

# Unlike R, you can actually save a lambda function in Python so that you can resuse it later?

# Let's say you have a list of numbers and want to create a new list containing the square of each element

data = [1, 2, 3, 4, 5]
new = []
for x in data:
    new.append(x**2)
    print(new)

# List comprehension in Python
data = [1, 2, 3, 4, 5]
new = [x**2 for x in data]
print(new)

# Dictionary comprehension

# Loop
data = [1, 2, 3, 4, 5]
new = {}
for x in data:
    new[x] = x ** 2
print(new)

# Comprehension
data = [1, 2, 3, 4, 5]
new = {x: x ** 2 for x in data}
print(new)

# map function python

# For loop
def sq(x):
    return x**2

l = [1,2, 3]
for i in l:
    print(sq(i))

# Map. Output of map is a mpap object
map(sq, l)

list(map(sq, l))

# List comprehension
# List comprehensions are a concise and convenient way to address a common programming task: iterating though a list, making a calculation, and saving the calculation into a new list. While this can be performed using a for loop, a list comprehension preforms the same task with fewer number of lines.

# The following list comprehension squares all values in a list:

x = [1, 2, 3, 4]
print([i**2 for i in x])

[1, 4, 9, 16]
A list of file names has been provided to you in the inflam_files list. Your task is to write a list comprehension that imports these files as pandas DataFrames in a single list.

# List comprehensions in Python are a convenient way to create a new list from an existing list! Now you will practice dictionary comprehensions.

# Append dataframes into list with for loop
dfs_list = []
for f in inflam_files:
    dat = pd.read_csv(f)
    dfs_list.append(dat)

# Re-write the provided for loop as a list comprehension: dfs_comp
dfs_comp = [pd.read_csv(f) for f in inflam_files]
print(dfs_comp)

import glob
dfs_list = glob.glob("/Users/dimitrispolychronopoulos/Documents/GEL_Tasks/SelectedFamilies/b37/*.csv")
dfs_comp = [pd.read_csv(f) for f in dfs_list]

# Dictionary comprehension
# A dictionary comprehension is very similar to a list comprehension. The difference is that the final result is a dictionary instead of a list. Recall that each element in a dictionary has 2 parts, a key and a value, which is seperated by a colon.

# The following dictionary comprehension squares all values in a list:

x = [['a', 1], ['b', 2], ['c', 3], ['d', 4]]
print({key:(value**2) for (key, value) in x})

{'b': 4, 'd': 16, 'c': 9, 'a': 1}
# Note:
# When you print a dictionary, the order of elements is not guaranteed.
# Dictionary comprehensions are wrapped inside { }.

# Inspect the contents of the 2D list twitter_followers in the shell.
# Write a dict comprehension where the key is first element of the sub-list, and the value is the second: tf_dict.
# Print tf_dict.

# Write a dict comprehension
tf_dict = {key:value for (key,value) in twitter_followers}

# Print tf_dict
print(tf_dict)

# Manually create a df by passing in a dic to the DataFrame() fun in pandas. Keys will be the colnames and the values will be the values within the column
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]},
     index = ['x', 'y', 'z'])

df[['A', 'B']]

# Subsetting rows
# 1. row-label (loc) vs row-index(iloc)
# 2. Python starts counting from 0

# Extract rows using row indices you need the iloc accessor

# this is to subset the first row of df
df.iloc[0]

# To select more than one row, you need to pass a list to .iloc
df.iloc[[0, 1]]


# You can also pass an optional colon after the comma to specify that you want to select all cols
df.iloc[0, :]

# Although if you don't specify the col, all cols are selected by def
df.iloc[[0,1], :]

# Extract rows using the row labels using the .loc accessor
df.loc['x']

df.loc[['x', 'y']]

# .loc and .iloc to subset both rows and cols simultaneously
df.loc['x', 'A']

df.loc[['x', 'y'], ['A', 'B']]

# Conditional subsetting
df[df.A == 3]

df[(df.A == 3) | (df.B == 4)]

# Similar to methods, you call an attribute on the object, but wo the parentheses

# Selecting rows and columns
# The syntax for subsetting both rows and columns in Pandas is similar to R. The rows and columns are specified to the left and right of the comma, respectively.

# To subset rows and columns using boolean expressions or labels, you can use:

# df.loc[row_labels, column_labels]
# To subset rows and columns using indices, you can use:

# df.iloc[row_indices, column_indices]

type(df)

# Info method shows the type of each column in the df

df.info()

# Change data type of cols by making use of the astype() method
# Object type in pandas refers to strings

# With the str accessor you can use Python's built-in string manipulation functions

df = pd.DataFrame({'name': ['Daniel ', ' Eric', ' Julia ']})

df['name_strip'] = df['name'].str.strip()

# Pandas also provides the category type which is analogous to factors in R
# Pass the string category to the astype function

df = pd.DataFrame({'name': ['Daniel', 'Eric', 'Julia'],
                   'gender': ['Male', 'Male', 'Female']})

df.dtypes

df['gender_cat'] = df['gender'].astype('category')

# See various categories using category accessor (cat accessor)

df['gender_cat'].cat.categories

df['gender_cat'].cat.codes

# to_datetime() fun from pandas to convert strings into dates
df = pd.DataFrame({'name': ['Rosaline Franklin', 'William Gosset'],
                   'born': ['1920-07-25', '1876-06-13']})

df['born_dt'] = pd.to_datetime(df['born'])
df.dtypes

# Access date components with the df accessor
df['born_dt'].dt.day

df['born_dt'].dt.month

df['born_dt'].dt.year

# Strings
# Columns containing strings are represented as the object type in Pandas.

# Since a lot of data you will encounter involve strings, it is important that you know how to manipulate them. Python allows you to use its built-in string manipulation methods with the str accessor. There are several string methods, some of which are .upper() and .lower(). They convert strings to upper and lower case, respectively.

# Converts 'col_a' to lower case
df['col_a'].str.lower()

# Converts 'col_b' to upper case
df['col_b'].str.upper()

# Category
# Pandas provides the category data type, which is analogous to the R factor.

# You can convert a column into a categorical data type by passing 'category' to the .astype() method. Once you have a categorical column, you can see the various categories (known as levels in R) by using the .cat accessor and calling the .categories attribute.

# Another use case for categorical values is when you want to preserve ordering in your data. For example, intuitively it makes sense that 'low' comes before 'high'. You can use reorder_categories() to provide an order to a column.

# Reorder categorical levels
df['column_name'].cat.reorder_categories(['low', 'high'], ordered=True)

import pandas as pd

# Load the country_timeseries dataset
ebola = pd.read_csv('country_timeseries.csv')

# Inspect the Date column
print(ebola['Date'].dtype)

# Convert the type of Date column into datetime
ebola['Date'] = pd.to_datetime(ebola.Date, format='%m/%d/%Y')

# Inspect the Date column
print(ebola['Date'].dtype)

# Dates (II)
# Instead of converting the type of a column after importing the data, you can import the data while parsing the dates correctly. To do this, you can pass the parse_dates argument of pd.read_csv() a list of column names that should be imported as dates. Once the date column is imported as the correct type (datetime64), you can make use of the .dt accessor along with the .year, .month, and .day attributes to can access the year, month, and day from these dates.

# Access year
df['Date'].dt.year

# Access month
df['Date'].dt.month

# Access day
df['Date'].dt.day

# RWD will always have missing data
# NaN missing values from from numpy
# np.NaN, np.NAN, np.nan are all the same as the NA R value

# Check non-missing with pd.notnull
# Check missing with pd.notnull
# pd.isnull is an alias for pd.isna

# Unlike R, missing vals are ignored by default when calculating the mean
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3],
                   'B': [4, 5, 6]})

# axis = 0 applies the fun column-wise which is the default and axis = 1 applies the function row-wise
df.apply(np.mean, axis = 0)

# groupby: split-apply-combine to perform calculations in your data

# Missing values
# It is very rare to find a dataset that doesn't contain any missing values. Missing values are represented as NaN in pandas. You can use the isnull() pandas function to check for missing values. pd.isnull(df['column']) will return True if the value is missing, or False if there are no missing values.

# Compared to R, missing values behave a little differently in Python. For example, the .mean() method automatically ignores missing values in Python. You can also recode missing values with the .fillna() method. This will replace all missing values in the column with the provided value.

# In this exercise, we've modified the tips dataset such that it contains some missing values.

# Print the rows where total_bill is missing
print(tips.loc[pd.isnull(tips['total_bill'])])

# Mean of the total_bill column
tbill_mean = tips['total_bill'].mean()

# Fill in missing total_bill
print(tips['total_bill'].fillna(tbill_mean))

# Great job! You can also drop missing values using the .dropna() method.

# Groupby
# Groupbys are an incredibly powerful and fast way to perform calculations, transformations, and filter your data. It follows the mantra of split-apply-combine where the dataset is split into multiple partitions based on unique values of a variable or variables, a function is applied on each partition separately, and all the results are combined together at the end.

# Mean tip by sex
print(tips.groupby('sex')['tip'].mean())

# Mean tip by sex and time
print(tips.groupby(['sex', 'time'])['tip'].mean())

# In addition to calculating the mean, you can use other methods such as .agg() and .filter() on grouped DataFrames.

# Tidy data
# Reshaping your data has several applications. One important application is to switch from a data analytics friendly to a reporting friendly format. This concept is further expanded in the Tidy data paper by Hadley Wickham.

# Data in a tidy format also allows you to perform groupby operations as seen in the previous exercise.

# In this exercise you will use melt() and .pivot_table() from pandas to reshape your data from one shape to another. Remember that when you call .pivot_table() on your data, you also need to call the .reset_index() method to get your original DataFrame back.

# Before you start reshaping the airquality DataFrame, inspect it in the shell. We've imported pandas as pd.

# Bivariate scatter plot.
iris.plot(kind='scatter', x='Sepal.Length', y = 'Sepal.Width')
plt.show()

# Boxplots
iris.boxplot(by='Species', column='Sepal.Length')
plt.show()

# Univariate plots in pandas
# You'll start this chapter by using the plotting methods in pandas. This is typically done by calling the .plot() method on the relevant column, and passing in a argument for the kind argument.

# Histograms and boxplots are good for continuous data.

# Histogram
df['column_name'].plot(kind='hist')
plt.show()

# Boxplot
df['column_name'].plot(kind='box')
plt.show()
# Bar graphs are good for categorical data. Remember to first obtain the counts of values using the .value_counts() method before plotting.

# Bar plot
counts = df['column_name'].value_counts()
counts.plot(kind='bar')
plt.show()
# Remember to use the show() function from matplotlib.pyplot to display the plot.

# Bivariate plots in pandas
# Comparing multiple variables simultaneously is also another useful way to understand your data. When you have two continuous variables, a scatter plot is usually used.

# Scatter plot
df.plot(x='x_column', y='y_column', kind='scatter')
plt.show()
# You can use a boxplot to compare one continuous and one categorical variable. However, you will be using the .boxplot() method instead of the .plot() method.

# Boxplot
df.boxplot(column='y_column', by='x_axis')
plt.show()

# Create a boxplot of the tip column grouped by the sex column.
import matplotlib.pyplot as plt

# Boxplot of the tip column by sex
tips.boxplot(column='tip', by='sex')
plt.show()

# Seaborn histograms
import seaborn as sns

# Seaborn is also built on matplotlib

# countplot function to draw a bar plot

sns.countplot('species', data = iris)

# Scatter plots with the regplot() function. Seaborn draws a regression line by default when plotting a scatter plot. You can turn it off as follows
sns.regplot(x='sepal_length', y='sepal_width', data = iris, fit_reg=False)

# You can specify the row and col arguments which can be used to create facets
sns.lmplot(x='sepal_length', y='sepal_width', data=iris, fit_reg=False, col='species')
plt.show()

# When using a plotting fun that does not support faceting, you can manually create a facet grid object
g = sns.FacetGrid(iris, col="species")
g = g.map(plt.hist, "sepal_length")
plt.show()

# Univariate plots in seaborn
# Seaborn is a popular plotting library. It embraces the concepts of "tidy data" and allows for quick ways to plot multiple varibles.

# You will begin by generating univariate plots. Barplots and histograms are created using the countplot() and distplot() functions, respectively.

import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot
sns.countplot(x='column_name', data=df)
plt.show()

# Histogram
sns.distplot(df['column_name'])
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of total_bill and tip faceted by smoker and colored by sex
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', col='smoker')
plt.show()

# Facet plots in seaborn
# Some plotting functions in seaborn such as distplot() and lmplot() have built-in facets. All you need to do is pass a col and/or row argument to create facets in your plot.

# For functions that do not have built-in facets, you can manually create them with the FacetGrid() function, and then specify the col and/or row to create your facets. To manually create a facetted plot, you can use the following code:

import seaborn as sns
import matplotlib.pyplot as plt

# Create a facet
facet = sns.FacetGrid(df, col='column_a', row='column_b')

# Generate a facetted scatter plot
facet.map(plt.scatter, 'column_x', 'column_y')
plt.show()
# You can add another layer of data to the plot by using the hue argument to color the points by a variable.

import seaborn as sns
import matplotlib.pyplot as plt

# FacetGrid of time and smoker colored by sex
facet = sns.FacetGrid(tips, col='time', row='smoker', hue='sex')

# Map the scatter plot of total_bill and tip to the FacetGrid
facet.map(plt.scatter, 'total_bill', 'tip')
plt.show()

# Matplotlib
# You can think of a figure as the entire image, and an axes is one subplot in that image. One axes means one subplot.  Notice the difference between axes and axis!

fig, ax = plt.subplots() # Create a single figure with one axes
ax.scatter(iris['sepal_length'], iris['sepal_width']) # ax is the axes object. We take it and draw a scatter plot on it
plt.show()

# How to create two axes in the figure with the subplots() function
fig, (ax1, ax2) = plt.subplots(1, 2) # This means figure with 1 row and 2 cols

# To prevent plots from overlapping on one another, you can use the clf() function to clear the figure

# Univariate and bivariate plots in matplotlib
# Matplotlib is at the core of all the plotting functions in Pandas and Seaborn. Understanding how matplotlib works will help you tweak your plot for publication or reports.

# In this exercise, you will generate a histogram and scatterplot:

# Histogram
plt.hist(df['column_name'])
plt.show()

# Scatter plot
plt.scatter(df['x_column'], df['y_column'])
plt.show()
Recall that you need to call plt.show() to display the plot.

# All the plotting methods (in Pandas) and functions (in Seaborn) you've seen so far build on top of Matplotlib. This is why you've been calling plt.show() for all the plotting exercises.

# Subfigures in matplotlib
# Sometimes you want to combine several subplots in a single figure.

# You can do this by creating a figure and an axes simultaneously by using the plot.subplots() function. This function returns the figure and the axes objects:

import matplotlib.pyplot as plt

# Create a figure and axes
# fig, (ax1, ax2) = plt.subplots(number_of_rows, number_of_columns)
# You can now use these axes objects to generate plots:

# Use the axes to plot visualizations
ax1.hist(df['column_1'])
ax2.hist(df['column_2'])
plt.show()

# Create a figure with 2 axes (1 row 2 columns).
# Generate a scatter plot in row 1 column 1, and a histogram in row 1, column 2.

import matplotlib.pyplot as plt

# Create a figure with scatter plot and histogram
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(tips['tip'], tips['total_bill'])
ax2.hist(tips['total_bill'])
plt.show()

# Working with axes
# Now that you've seen how to create a figure within matplotlib, you can extend this knowledge to the other plotting libraries you've used. If you look at what gets returned from the various seaborn plotting calls, most functions will return an axes, while some will return a figure. You can leverage this information when creating axes' in a figure. This way you are not limited to just using base matplotlib as a plotting library when making more complex figures.

# Create a figure with 2 axes (1 row 2 columns).
# Generate a histogram in row 1 column 1 and a scatter plot in row 1 column 2.

# Figure with 2 axes: regplot and distplot
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(tips['tip'], ax=ax1)
sns.regplot(x='total_bill', y='tip', data=tips, ax=ax2)
plt.show()

# Polishing up a figure
# Since plt.subplots() returns axes object, you can use the .set_title(), .set_xlabel(), and .set_ylabel() methods to add labels to your figure.

# Use the .set_title() method on the axes to label the plot: 'Histogram'.
# Use the .set_xlabel() method on the axes to label the plot: 'Total Bill'

import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with 1 axes
fig, ax = plt.subplots()

# Draw a displot
ax = sns.distplot(tips['total_bill'])

# Label the title and x axis
ax.set_title('Histogram')
ax.set_xlabel('Total Bill')
plt.show()

# List comprehensions to load all the data
# Saves repetitive typing

# glob() from the glob module to get a list of all csv files in the current directory
import glob
import pandas as pd

scv_files = glob.glob('*.csv')

csv_files

all_dfs = [pd.read_csv(x) for x in csv_files]

all_dfs[0]

# Load multiple data files
# It's perfectly fine to manually import multiple datasets. However, there will be times when you'd want to import a bunch of datasets without having to make multiple read_csv() calls. You can use the glob library that is built into Python to look for files that match a pattern. The library is called "glob" because "globbing" is the way patterns are specified in the Bash shell.

# The glob() function returns a list of filenames that match a specified pattern. You can then use a list comprehension to import multiple files into a list, and then you can extract the DataFrame of interest.

# Obtain a list of all csv files in your current directory and assign it to csv_files.
# Write a list comprehension that reads in all the csv files into a list, dfs.
# Write a list comprehension that looks at the .shape of each DataFrame in the list.

import glob
import pandas as pd

# Get a list of all the csv files
csv_files = glob.glob('*.csv')

# List comprehension that loads of all the files
dfs = [pd.read_csv(x) for x in csv_files]

# List comprehension that looks at the shape of all DataFrames
print([x.shape for x in dfs])

# You could also have used a dictionary comprehension instead of a list comprehension to import your data.

# Explore
# Now that you have a list of DataFrames, let's explore one of them, the planes dataset. Since you have a list of file names and a corresponding list of DataFrames, you can determine that the planes dataset is the third element in dfs.

# Recall that you can call .value_counts() method to get a frequency count of the unique values in a column, and use .loc accessor to use boolean subsetting to filter the data. When combining multiple boolean expressions, the & and | operators stand for 'and', and 'or', respectively. Also, remember the boolean expressions need to be wrapped around a pair of parentheses.

# Get the planes DataFrame
planes = dfs[2]

# Count the frequency of engines in our data
print(planes['engines'].value_counts())

# Look at all planes with >= 3 engines
print(planes.loc[planes['engines'] >= 3])

# Look at all planes with >= 3 engines and < 100 seats
print(planes.loc[(planes['engines'] >= 3) & (planes['seats'] <= 100)])

# Visualize
# It's interesting to see that there are so few planes with three or more engines and 100 or less seats. What is even more interesting is that there is a plane with 4 engines and 2 seats. Turns out this is a fighter jet.

# If you plotted this data, you would quickly be able to see how common or unusual data are. Here you'll use the pandas plotting methods to do a quick exploratory plot.

# Instead of writing multiple groupby statements, you can use the aggregate method and pass a list of funs such as mean and max

df_melt.groupby('name')['value'].agg(['mean', 'max'])

# categorical variables in your data are recoded into dummy variables, that is they are recoded as integers. This is also called one-hot encoding

import numpy as np

df = pd.DataFrame({
    'status': ['sick', 'healthy', 'sick'],
    'treatment_a': [np.NaN, 16, 3],
    'treatment_b': [2, 11, 1]
})

# Get Dummies function from Pandas will return a new df where the non_numerical columns will be encoded as dummy variables
pd.get_dummies(df)

# Recode dates
# Now let's look at another dataset in the nycflights13 datasets, specifically the flights dataset. This dataset gives us the time when a particular flight departed from one of the New York City airports (JFK, LGA, EWR), it's scheduled departure and arrival times, and the departure and arrival delays. You'll begin by investigating to determine if there are any seasonal patterns for delays.

# Here are the cutoff times for each season in the United States.

# SPRING EQUINOX	March 20
# SUMMER SOLSTICE	June 21
# FALL EQUINOX	September 22
# WINTER SOLSTICE	December 21
# We defined the get_season() function that converts a given date to a season (one of winter, spring, summer, and fall).

# Groupby aggregates
# Now let's look for some patterns in the data. One way you can do this is to look at groupby aggregates of the data. You will see if delays vary with season, carrier and origin airport.

# You will visualize this data in the next exercise and in order for the plots to render nicely, we will use a subset of the original flights data in these exercises.

# Calculate total_delay
flights['total_delay'] = flights['dep_delay'] + flights['arr_delay']

# Mean total_delay by carrier
tdel_car = flights.groupby('carrier')['total_delay'].mean().reset_index()
print(tdel_car)

# Calculate total_delay
flights['total_delay'] = flights['dep_delay'] + flights['arr_delay']

# Mean total_delay by carrier
tdel_car = flights.groupby('carrier')['total_delay'].mean().reset_index()
print(tdel_car)

# Mean dep_delay and arr_delay for each season
dadel_season = flights.groupby('season')['dep_delay', 'arr_delay'].mean().reset_index()
print(dadel_season)

# Calculate total_delay
flights['total_delay'] = flights['dep_delay'] + flights['arr_delay']

# Mean total_delay by carrier
tdel_car = flights.groupby('carrier')['total_delay'].mean().reset_index()
print(tdel_car)

# Mean dep_delay and arr_delay for each season
dadel_season = flights.groupby('season')['dep_delay', 'arr_delay'].mean().reset_index()
print(dadel_season)

# Mean and std delays by origin
del_ori = flights.groupby('origin')['total_delay', 'dep_delay', 'arr_delay'].agg(['mean', 'std'])
print(del_ori)

# Create a figure with two axes (2 rows and 1 column).
# Use seaborn and flights to create a boxplot of 'origin' and 'dep_delay' on x and y axes, respectively.
# Use seaborn and tdel_car to create a bar plot of 'carrier' and 'total_delay' on x and y axes, respectively.
# Label the boxplot: 'Originating airport and the departure delay'.

# Create a figure
fig, (ax1, ax2) = plt.subplots(2, 1)

# Boxplot and barplot in the axes
sns.boxplot(x='origin', y='dep_delay', data=flights, ax=ax1)
sns.barplot(x='carrier', y='total_delay', data=tdel_car, ax=ax2)

# Label axes
ax1.set_title('Originating airport and the departure delay')

# Use tight_layout() so the plots don't overlap
fig.tight_layout()
plt.show()

# Dummy variables
# In the last exercise of the course, you will prepare your data for modeling by dummy encoding your non-numeric columns. For example, if you have a column of gender values, 'Male' and 'Female', you want separate columns that tell you whether the observation is from a 'Male' or a 'Female'. This process of creating dummy variables is also called one-hot encoding.

# You can use the get_dummies() function from pandas to convert the non-numeric columns into dummy variables.

df_new = pd.get_dummies(df)
# We've subsetted the flights DataFrame to create flights_sub to make it easier to see what is happening.

# Great! Did you notice that the origin column is converted into three columns consisting of 0s and 1s? Creating dummy variables is one of the steps you'll be taking before using other Python libraries such as scikit-learn to fit a model to your data.

# Raw strings

colors = {'crimson': 0xdc143c, 'coral': 0xdff7f50, 'teal': 0x008080}
for color in colors:
    print(color, colors[color])


path = "/Users/dimitrispolychronopoulos/Documents/GEL_Tasks/FamSelectionOutput"
os.chdir(path)

filenames = ['RD37_fam_selection_output_2019_01_16_100224.csv', 'RD37_fam_selection_output_2019_01_22_194405.csv']
dataframes = []
for f in filenames:
    dataframes.append(pd.read_csv(f))

# or with a list comprehension
filenames = ['RD37_fam_selection_output_2019_01_16_100224.csv', 'RD37_fam_selection_output_2019_01_22_194405.csv']
dataframes = [pd.read_csv(f) for f in filenames]

from glob import glob
filenames = glob('RD37*.csv')

