# Exploring the NSFG data
# To get the number of rows and columns in a DataFrame, you can read its shape attribute.
#
# To get the column names, you can read the columns attribute. The result is an Index, which is a Pandas data structure that is similar to a list. Let's begin exploring the NSFG data! It has been pre-loaded for you into a DataFrame called nsfg.

# value_counts() to see what values appear in pounds and how many times each value appears. By default results are sorted with most freq value first so using sort_index() sorts them by value instead

# describe() computes summary stats

# replace takes a list of values we want to replace and the value we want to replace them with

pounds = pounds.replace([98, 99], np.nan) # np.nan means we are getting the special value NaN from the NumPy library

# Clean a variable
# In the NSFG dataset, the variable 'nbrnaliv' records the number of babies born alive at the end of a pregnancy.
#
# If you use .value_counts() to view the responses, you'll see that the value 8 appears once, and if you consult the codebook, you'll see that this value indicates that the respondent refused to answer the question.
#
# Your job in this exercise is to replace this value with np.nan. Recall from the video how Allen replaced the values 98 and 99 in the ounces column using the .replace() method:
#
# ounces.replace([98, 99], np.nan, inplace=True)

# In the 'nbrnaliv' column, replace the value 8, in place, with the special value NaN.
# Confirm that the value 8 no longer appears in this column by printing the values and their frequencies.

# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace([8], np.nan, inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())

# Compute a variable
# For each pregnancy in the NSFG dataset, the variable 'agecon' encodes the respondent's age at conception, and 'agepreg' the respondent's age at the end of the pregnancy.
#
# Both variables are recorded as integers with two implicit decimal places, so the value 2575 means that the respondent's age was 25.75.

# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())

# Good job. A variable that's computed from other variables is sometimes called a 'recode'. It's now time to get back to the motivating question for this chapter: what is the average birth weight for babies in the U.S.? See you in the next video!

# ~ operator is logical NOT or inverse, for instance: full_term_weight = birth_weight[~preterm]

# Make a histogram
# Histograms are one of the most useful tools in exploratory data analysis. They quickly give you an overview of the distribution of a variable, that is, what values the variable can have, and how many times each value appears.
#
# As we saw in a previous exercise, the NSFG dataset includes a variable 'agecon' that records age at conception for each pregnancy. Here, you're going to plot a histogram of this variable. You'll use the bins parameter that you saw in the video, and also a new parameter - histtype - which you can read more about here in the matplotlib documentation. Learning how to read documentation is an essential skill. If you want to learn more about matplotlib, you can check out DataCamp's Introduction to Matplotlib course.

# Plot the histogram
plt.hist(agecon, bins=20, histtype = 'step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

# Compute birth weight
# Now let's pull together the steps in this chapter to compute the average birth weight for full-term babies.
#
# I've provided a function, resample_rows_weighted, that takes the NSFG data and resamples using the sampling weights in wgt2013_2015. The result is a DataFrame, sample, that is representative of the U.S. population.
#
# Then I extract birthwgt_lb1 and birthwgt_oz1, replacing special codes with NaN:
#
# # Resample the data
# sample = resample_rows_weighted(nsfg, 'wgt2013_2015')
#
# # Clean the weight variables
# pounds = sample['birthwgt_lb1'].replace([98, 99], np.nan)
# ounces = sample['birthwgt_oz1'].replace([98, 99], np.nan)

# Use pounds and ounces to compute total birth weight.
# Make a Boolean Series called preterm that is true for babies with 'prglngth' less than 37 weeks.
# Use preterm to select birth weight for babies that are not preterm. Store the result in full_term_weight.
# Compute the mean weight of full term babies.

# Compute total birth weight
birth_weight = pounds + ounces/16

# Create a Boolean Series for preterm babies
preterm = sample['prglngth'] < 37

# Select the weights of full term babies
full_term_weight = birth_weight[~preterm]

# Compute the mean weight of full term babies
print(np.mean(full_term_weight))

# Filter
# In the previous exercise, you computed the mean birth weight for full-term babies; you filtered out preterm babies because their distribution of weight is different.
#
# The distribution of weight is also different for multiple births, like twins and triplets. In this exercise, you'll filter them out, too, and see what effect it has on the mean.

# Use the variable 'nbrnaliv' to make a Boolean Series that is True for single births (where 'nbrnaliv' equals 1) and False otherwise.
# Use Boolean Series and logical operators to select single, full-term babies and compute their mean birth weight.
# For comparison, select multiple, full-term babies and compute their mean birth weight.

# Filter preterm babies
preterm = sample['prglngth'] < 37

# Filter single births
single = sample['nbrnaliv'] == 1

# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[~preterm & single]
print('Single full-term mean:', single_full_term_weight.mean())

# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[~preterm & ~single]
print('Multiple full-term mean:', mult_full_term_weight.mean())

plt.hist(educ.dropna(), label='educ')

plt.show()

# PMF or probabilty Mass Function that contains the unique values in the dataset and how often each one appears

# Make a PMF
# The GSS dataset has been pre-loaded for you into a DataFrame called gss. You can explore it in the IPython Shell to get familiar with it.
#
# In this exercise, you'll focus on one variable in this dataset, 'year', which represents the year each respondent was interviewed.

# Plot a PMF
# Now let's plot a PMF for the age of the respondents in the GSS dataset. The variable 'age' contains respondents' age in years.

# Select the age column
age = gss['age']

# Make a PMF of age
pmf_age = Pmf(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()

# PMF represents the possible values in a distribution and their probabilities
# CDF or Cumulative Distribution Functions; great way to visualise and compare distributions

# From PMF to CDF
# If you draw a random element from a distribution:

# - PMF (Probability Mass Function) is the probability that you get exactly x
# - CDF (Cumulative Distribution Function) is the probability you get a value   <= x for a given value of x

# CDF is an invertible function which means that if u have a prob p you can look up the corresponding quantity q

# The distance from the 25th to the 75th percentile is called the IQR. It measures the spread of the distribution so it is similar to sd or variance

# IQR can be more robust than variance

# Compute IQR
# Recall from the video that the interquartile range (IQR) is the difference between the 75th and 25th percentiles. It is a measure of variability that is robust in the presence of errors or extreme values.
#
# In this exercise, you'll compute the interquartile range of income in the GSS dataset. Income is stored in the 'realinc' column, and the CDF of income has already been computed and stored in cdf_income.

# Calculate the 75th percentile
percentile_75th = cdf_income.inverse(0.75)

# Calculate the 25th percentile
percentile_25th = cdf_income.inverse(0.25)

# Calculate the interquartile range
iqr = percentile_75th - percentile_25th

# Print the interquartile range
print(iqr)

# Plot a CDF
# The distribution of income in almost every country is long-tailed; that is, there are a small number of people with very high incomes.
#
# In the GSS dataset, the variable 'realinc' represents total household income, converted to 1986 dollars. We can get a sense of the shape of this distribution by plotting the CDF.

# Select realinc
income = gss['realinc']

# Make the CDF
cdf_income = Cdf(income)

# Plot it
cdf_income.plot()

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.show()

# Comparing distributionsm, in general CDFs are smoother thatn PMFs

# Extract education levels
# Let's create Boolean Series to identify respondents with different levels of education.
#
# In the U.S, 12 years of education usually means the respondent has completed high school (secondary education). A respondent with 14 years of education has probably completed an associate degree (two years of college); someone with 16 years has probably completed a bachelor's degree (four years of college).

# Complete the line that identifies respondents with associate degrees, that is, people with more than 14 years of education but less than 16.
# Complete the line that identifies respondents with 12 or fewer years of education.
# Confirm that the mean of high is the fraction we computed in the previous exercise, about 53%.

# Plot income CDFs
# Let's now see what the distribution of income looks like for people with different education levels. You can do this by plotting the CDFs. Recall how Allen plotted the income CDFs of respondents interviewed before and after 1995:
#
# Cdf(income[pre95]).plot(label='Before 1995')
# Cdf(income[~pre95]).plot(label='After 1995')
# You can assume that Boolean Series have been defined, as in the previous exercise, to identify respondents with different education levels: high, assc, bach, and post.

income = gss['realinc']

# Plot the CDFs
Cdf(income[high]).plot(label='High school')
Cdf(income[assc]).plot(label='Associate')
Cdf(income[bach]).plot(label='Bachelor')
Cdf(income[post]).plot(label='Postgraduate')

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.legend()
plt.show()

# Estimate PDFs from data
# SciPy provides an object called norm that represents the normal distribution


sample = np.random.normal(size=1000)

sample

# Create an array of equally spaced points from -3 to 3
import numpy as np
xs = np.linspace(-3, 3)

# norm(0,1) creates an obj that represents a normal distribution with mean 0 and sd 1. .cdf evaluates the cdf of the normal dist
ys = norm(0,1).cdf(xs)

import matplotlib.pyplot as plt

plt.plot(xs, ys, color = 'gray')

# Plotting the cdf of the normal distributon and actual data

ys = norm(0,1).pdf(xs) # evaluates the PDF (probability density function)

plt.plot(xs, ys, color = 'gray')

# Kernel Density Estimation: A way of getting from a PMF to a PDF

import seaborn as sns

# kdeplot from seaborn takes the sample, estimates the pdf and then plots it
sns.kdeplot(sample)

# compare the KDE plot with the normal PDF

# 3 ways to visualise distributions:
# CDFs: for exploration
# PMFs if there are a small number of unique values
# KDE if there are a lot of values

# Distribution of income
# In many datasets, the distribution of income is approximately lognormal, which means that the logarithms of the incomes fit a normal distribution. We'll see whether that's true for the GSS data. As a first step, you'll compute the mean and standard deviation of the log of incomes using NumPy's np.log10() function.
#
# Then, you'll use the computed mean and standard deviation to make a norm object using the scipy.stats.norm() function.

# Extract 'realinc' from gss and compute its logarithm using np.log10().
# Compute the mean and standard deviation of the result.
# Make a norm object by passing the computed mean and standard deviation to norm().

# Extract realinc and compute its log
income = gss['realinc']
log_income = np.log10(income)

# Compute mean and standard deviation
mean = log_income.mean()
std = log_income.std()
print(mean, std)

# Make a norm object
from scipy.stats import norm
dist = norm(mean, std)

# Comparing CDFs
# To see whether the distribution of income is well modeled by a lognormal distribution, we'll compare the CDF of the logarithm of the data to the CDF of a normal distribution with the same mean and standard deviation. The dist object you created in the previous exercise is available for use:
#
# from scipy.stats import norm
# dist = norm(mean, std)
# This is a norm object with the same mean and standard deviation as the data. All scipy.stats.norm objects have a .cdf() method.

# Evaluate the normal CDF using dist and xs.
# Plot the CDF of the logarithms of the incomes, using log_income, which is a Series object.

# Evaluate the normal CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data CDF
Cdf(log_income).plot()

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()

# Comparing PDFs
# In the previous exercise, we used CDFs to see if the distribution of income is lognormal. We can make the same comparison using a PDF and KDE. That's what you'll do in this exercise!
#
# As before, the norm object dist is available in your workspace:
#
# from scipy.stats import norm
# dist = norm(mean, std)
# Just as all norm objects have a .cdf() method, they also have a .pdf() method.
#
# To create a KDE plot, you can use Seaborn's kdeplot() function. To learn more about this function and Seaborn, you can check out DataCamp's Data Visualization with Seaborn course. Here, Seaborn has been imported for you as sns.

# Evaluate the normal PDF using dist, which is a norm object with the same mean and standard deviation as the data.
# Make a KDE plot of the logarithms of the incomes, using log_income, which is a Series object.

# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income)

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show()

# Exploring relationships
# jittering: adding random noise to data

# PMF of age
# Do people tend to gain weight as they get older? We can answer this question by visualizing the relationship between weight and age. But before we make a scatter plot, it is a good idea to visualize distributions one variable at a time. Here, you'll visualize age using a bar chart first. Recall that all PMF objects have a .bar() method to make a bar chart.
#
# The BRFSS dataset includes a variable, 'AGE' (note the capitalization!), which represents each respondent's age. To protect respondents' privacy, ages are rounded off into 5-year bins. 'AGE' contains the midpoint of the bins.

# Extract the variable 'AGE' from the DataFrame brfss and assign it to age.
# Plot the PMF of age as a bar chart.

# Extract age
age = brfss['AGE']

# Plot the PMF
Pmf(age).bar()

# Label the axes
plt.xlabel('Age in years')
plt.ylabel('PMF')
plt.show()

# Scatter plot
# Now let's make a scatterplot of weight versus age. To make the code run faster, I've selected only the first 1000 rows from the brfss DataFrame.
#
# weight and age have already been extracted for you. Your job is to use plt.plot() to make a scatter plot.
#

# Make a scatter plot of weight and age with format string 'o' and alpha=0.1.

# Select the first 1000 respondents
brfss = brfss[:1000]

# Extract age and weight
age = brfss['AGE']
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', alpha=0.1)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')

plt.show()

# So far so good. By adjusting alpha we can avoid saturating the plot. Next we'll jitter the data to break up the columns.

# Jittering
# In the previous exercise, the ages fall in columns because they've been rounded into 5-year bins. If we jitter them, the scatter plot will show the relationship more clearly. Recall how Allen jittered height and weight in the video:
#
# height_jitter = height + np.random.normal(0, 2, size=len(brfss))
# weight_jitter = weight + np.random.normal(0, 2, size=len(brfss))

# Add random noise to age with mean 0 and standard deviation 2.5.
# Make a scatter plot between weight and age with marker size 5 and alpha=0.2. Be sure to also specify 'o'.

# Select the first 1000 respondents
brfss = brfss[:1000]

# Add jittering to age
age = brfss['AGE'] + np.random.normal(0, 2.5, size=len(brfss))
# Extract weight
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight,  'o', markersize = 5, alpha = 0.2)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')
plt.show()

# Seaborn provides a function that makes violin plots but before using it we have to get rid of any rows with missing data
# data = df.dropna(subset = ['AGE', 'WTKG3'])
sns.violinplot(x = 'AGE', y = 'WTKG3', data = data, inner = None) # inner None simplifies the plot a little bit
plt.show()

sns.boxplot(x = 'AGE', y= 'WTKG3', data=data, whis=10) #whis = 10 to turn off a feature we don't need
# Each box represent the IQR from the 25th to the 75th percentile

# Height and weight
# Previously we looked at a scatter plot of height and weight, and saw that taller people tend to be heavier. Now let's take a closer look using a box plot. The brfss DataFrame contains a variable '_HTMG10' that represents height in centimeters, binned into 10 cm groups.
#
# Recall how Allen created the box plot of 'AGE' and 'WTKG3' in the video, with the y-axis on a logarithmic scale:
#
# sns.boxplot(x='AGE', y='WTKG3', data=data, whis=10)
# plt.yscale('log')

# Fill in the parameters of .boxplot() to plot the distribution of weight ('WTKG3') in each height ('_HTMG10') group. Specify whis=10, just as was done in the video.
# Add a line to plot the y-axis on a logarithmic scale.

# Drop rows with missing data
data = brfss.dropna(subset=['_HTMG10', 'WTKG3'])

# Make a box plot
sns.boxplot(x = '_HTMG10', y = 'WTKG3', data = data, whis=10)

# Plot the y-axis on a log scale
plt.yscale('log')

# Remove unneded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

# Distribution of income
# In the next two exercises we'll look at relationships between income and other variables. In the BRFSS, income is represented as a categorical variable; that is, respondents are assigned to one of 8 income categories. The variable name is 'INCOME2'. Before we connect income with anything else, let's look at the distribution by computing the PMF. Recall that all Pmf objects have a .bar() method.

# Extract 'INCOME2' from the brfss DataFrame and assign it to income.
# Plot the PMF of income as a bar chart.

# Extract income
income = brfss['INCOME2']

# Plot the PMF
Pmf(income).bar()

# Label the axes
plt.xlabel('Income level')
plt.ylabel('PMF')
plt.show()

# Good work. Almost half of the respondents are in the top income category, so this dataset doesn't distinguish between the highest incomes and the median. But maybe it can tell us something about people with incomes below the median.

# Income and height
# Let's now use a violin plot to visualize the relationship between income and height.

# Create a violin plot to plot the distribution of height ('HTM4') in each income ('INCOME2') group. Specify inner=None to simplify the plot.

# Drop rows with missing data
data = brfss.dropna(subset=['INCOME2', 'HTM4'])

# Make a violin plot
sns.violinplot(x = 'INCOME2', y = 'HTM4', data = data, inner = None)

# Remove unneded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Income level')
plt.ylabel('Height in cm')
plt.show()

# If correlation is high, that is close to 1 or -1 you can conclude that there is a strong linear relationship

# But if the correlation is close to 0 that does not mean there is no relationship; there might be a strong non-linear relationship!

# Computing correlations
# The purpose of the BRFSS is to explore health risk factors, so it includes questions about diet. The variable '_VEGESU1' represents the number of servings of vegetables respondents reported eating per day.
#
# Let's see how this variable relates to age and income.

# From the brfss DataFrame, select the columns 'AGE', 'INCOME2', and '_VEGESU1'.
# Compute the correlation matrix for these variables.

# Select columns
columns = ['AGE', 'INCOME2', '_VEGESU1']
subset = brfss[columns]

# Compute the correlation matrix
print(subset.corr())

# Income and vegetables
# As we saw in a previous exercise, the variable '_VEGESU1' represents the number of vegetable servings respondents reported eating per day.
#
# Let's estimate the slope of the relationship between vegetable consumption and income.

# Extract the columns 'INCOME2' and '_VEGESU1' from subset into xs and ys respectively.
# Compute the simple linear regression of these variables.

from scipy.stats import linregress

# Extract the variables
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']

# Compute the linear regression
res = linregress(xs, ys)
print(res)

# Fit a line
# Continuing from the previous exercise:
#
# Assume that xs and ys contain income codes and daily vegetable consumption, respectively, and
#
# res contains the results of a simple linear regression of ys onto xs.
#
# Now, you're going to compute the line of best fit. NumPy has been imported for you as np.

# Set fx to the minimum and maximum of xs, stored in a NumPy array.
# Set fy to the points on the fitted line that correspond to the xs.

# Plot the scatter plot
plt.clf()
x_jitter = xs + np.random.normal(0, 0.15, len(xs))
plt.plot(x_jitter, ys, 'o', alpha=0.2)

# Plot the line of best fit
fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-', alpha=0.7)

plt.xlabel('Income code')
plt.ylabel('Vegetable servings per day')
plt.ylim([0, 6])
plt.show()

# Using StatsModels
# Let's run the same regression using SciPy and StatsModels, and confirm we get the same results.

# Compute the regression between 'INCOME2' and '_VEGESU1' using SciPy's linregress().
# Compute the regression between 'INCOME2' and '_VEGESU1' using StatsModels, with '_VEGESU1' as the intercept.

from scipy.stats import linregress
import statsmodels.formula.api as smf

# Run regression with linregress
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']
res = linregress(xs, ys)
print(res)

# Run regression with StatsModels
results = smf.ols('_VEGESU1 ~ INCOME2', data = brfss).fit()
print(results.params)

# Correlation and simple regression can't measure non-linear relationships but multiple regression can

# Plot income and edcuation
# To get a closer look at the relationship between income and education, let's use the variable 'educ' to group the data, then plot mean income in each group.
#
# Here, the GSS dataset has been pre-loaded into a DataFrame called gss.

# Group gss by 'educ'. Store the result in grouped.
# From grouped, extract 'realinc' and compute the mean.
# Plot mean_income_by_educ as a scatter plot. Specify 'o' and alpha=0.5.

# Group by educ
grouped = gss.groupby('educ')

# Compute mean income in each group
mean_income_by_educ = grouped['realinc'].mean()

# Plot mean income as a scatter plot
plt.plot(mean_income_by_educ, 'o', alpha = 0.5)

# Label the axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.show()

# Well done. It looks like the relationship between income and education is non-linear.

# Non-linear model of education
# The graph in the previous exercise suggests that the relationship between income and education is non-linear. So let's try fitting a non-linear model.

# Add a column named 'educ2' to the gss DataFrame; it should contain the values from 'educ' squared.
# Run a regression model that uses 'educ', 'educ2', 'age', and 'age2' to predict 'realinc'.

import statsmodels.formula.api as smf

# Add a new column with educ squared
gss['educ2'] = gss['educ'] ** 2

# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 + age + age2', data = gss).fit()

# Print the estimated parameters
print(results.params)

import pandas as pd
df = pd.DataFrame()

df['age'] = np.linspace(18, 85)
df['age2'] = df['age'] ** 2
df['educ'] = 12
df['educ2'] = df['educ'] ** 2

# Making predictions
# At this point, we have a model that predicts income using age, education, and sex.
#
# Let's see what it predicts for different levels of education, holding age constant.

# Using np.linspace(), add a variable named 'educ' to df with a range of values from 0 to 20.
# Add a variable named 'age' with the constant value 30.
# Use df to generate predicted income as a function of education.

# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()

# Make the DataFrame
df = pd.DataFrame()
df['educ'] = np.linspace(0, 20)
df['age'] = 30
df['educ2'] = df['educ']**2
df['age2'] = df['age']**2

# Generate and plot the predictions
pred = results.predict(df)
print(pred.head())

# Visualizing predictions
# Now let's visualize the results from the previous exercise!

# Plot mean_income_by_educ using circles ('o'). Specify an alpha of 0.5.
# Plot the prediction results with a line, with df['educ'] on the x-axis and pred on the y-axis.

# Plot mean income in each age group
plt.clf()
grouped = gss.groupby('educ')
mean_income_by_educ = grouped['realinc'].mean()
plt.plot(mean_income_by_educ, 'o', alpha = 0.5)

# Plot the predictions
pred = results.predict(df)
plt.plot(df['educ'], pred, label='Age 30')

# Label axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.legend()
plt.show()

# Logistic regression: categorical variables

# Parameters of a logistic regression are in the form of log odds

# Positive values are associated with things that make the outcome more likely; (-) values make the outcome less likely

# Log regression is a powerful tool for exploring relationships between a binary variable and the factors that predict it

# Predicting a binary variable
# Let's use logistic regression to predict a binary variable. Specifically, we'll use age, sex, and education level to predict support for legalizing cannabis (marijuana) in the U.S.
#
# In the GSS dataset, the variable grass records the answer to the question "Do you think the use of marijuana should be made legal or not?"

# Recode grass
gss['grass'].replace(2, 0, inplace=True)

# Run logistic regression
results = smf.logit('grass ~ age + age2 + educ + educ2 + C(sex)', data=gss).fit()
results.params

# Make a DataFrame with a range of ages
df = pd.DataFrame()
df['age'] = np.linspace(18, 89)
df['age2'] = df['age']**2

# Set the education level to 12
df['educ'] = 12
df['educ2'] = df['educ']**2

# Generate predictions for men and women
df['sex'] = 1
pred1 = results.predict(df)

df['sex'] = 2
pred2 = results.predict(df)

plt.clf()
grouped = gss.groupby('age')
favor_by_age = grouped['grass'].mean()
plt.plot(favor_by_age, 'o', alpha=0.5)

plt.plot(df['age'], pred1, label='Male')
plt.plot(df['age'], pred2, label = 'Female')

plt.xlabel('Age')
plt.ylabel('Probability of favoring legalization')
plt.legend()
plt.show()

