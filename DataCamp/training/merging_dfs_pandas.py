# Reindexing DataFrame from a list
# Sorting methods are not the only way to change DataFrame Indexes. There is also the .reindex() method.
#
# In this exercise, you'll reindex a DataFrame of quarterly-sampled mean temperature values to contain monthly samples (this is an example of upsampling or increasing the rate of samples, which you may recall from the pandas Foundations course).
#
# The original data has the first month's abbreviation of the quarter (three-month interval) on the Index, namely Apr, Jan, Jul, and Oct. This data has been loaded into a DataFrame called weather1 and has been printed in its entirety in the IPython Shell. Notice it has only four rows (corresponding to the first month of each quarter) and that the rows are not sorted chronologically.
#
# You'll initially use a list of all twelve month abbreviations and subsequently apply the .ffill() method to forward-fill the null entries when upsampling. This list of month abbreviations has been pre-loaded as year.

# Reorder the rows of weather1 using the .reindex() method with the list year as the argument, which contains the abbreviations for each month.
# Reorder the rows of weather1 just as you did above, this time chaining the .ffill() method to replace the null values with the last preceding non-null value.

# Reindexing using another DataFrame Index
# Another common technique is to reindex a DataFrame using the Index of another DataFrame. The DataFrame .reindex() method can accept the Index of a DataFrame or Series as input. You can access the Index of a DataFrame with its .index attribute.
#
# The Baby Names Dataset from data.gov summarizes counts of names (with genders) from births registered in the US since 1881. In this exercise, you will start with two baby-names DataFrames names_1981 and names_1881 loaded for you.
#
# The DataFrames names_1981 and names_1881 both have a MultiIndex with levels name and gender giving unique labels to counts in each row. If you're interested in seeing how the MultiIndexes were set up, names_1981 and names_1881 were read in using the following commands:
#
# names_1981 = pd.read_csv('names1981.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# names_1881 = pd.read_csv('names1881.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# As you can see by looking at their shapes, which have been printed in the IPython Shell, the DataFrame corresponding to 1981 births is much larger, reflecting the greater diversity of names in 1981 as compared to 1881.
#
# Your job here is to use the DataFrame .reindex() and .dropna() methods to make a DataFrame common_names counting names from 1881 that were still popular in 1981.

# Create a new DataFrame common_names by reindexing names_1981 using the Index of the DataFrame names_1881 of older names.
# Print the shape of the new common_names DataFrame. This has been done for you. It should be the same as that of names_1881.
# Drop the rows of common_names that have null counts using the .dropna() method. These rows correspond to names that fell out of fashion between 1881 & 1981.
# Print the shape of the reassigned common_names DataFrame. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)

# Scalar multiplication using asterisk
# weather.loc['2013-07-01':'2013-07-07', 'PrecipitationIn'] * 2.54

# df divide option with the option axis = 'rows'
week1_range.divide(week1_mean, axis='rows')

# Percentage changes using pct_change()

# Add dataframes: bronze + silver
bronze.add(silver)

# modify this behavior using fill_value = 0
# Chaining .add()

bronze.add(silver, fill_value=0).add(gold, fill_value = 0)

# Broadcasting in arithmetic formulas
# In this exercise, you'll work with weather data pulled from wunderground.com. The DataFrame weather has been pre-loaded along with pandas as pd. It has 365 rows (observed each day of the year 2013 in Pittsburgh, PA) and 22 columns reflecting different weather measurements each day.
#
# You'll subset a collection of columns related to temperature measurements in degrees Fahrenheit, convert them to degrees Celsius, and relabel the columns of the new DataFrame to reflect the change of units.
#
# Remember, ordinary arithmetic operators (like +, -, *, and /) broadcast scalar values to conforming DataFrames when combining scalars & DataFrames in arithmetic expressions. Broadcasting also works with pandas Series and NumPy arrays.

# Create a new DataFrame temps_f by extracting the columns 'Min TemperatureF', 'Mean TemperatureF', & 'Max TemperatureF' from weather as a new DataFrame temps_f. To do this, pass the relevant columns as a list to weather[].
# Create a new DataFrame temps_c from temps_f using the formula (temps_f - 32) * 5/9.
# Rename the columns of temps_c to replace 'F' with 'C' using the .str.replace('F', 'C') method on temps_c.columns.
# Print the first 5 rows of DataFrame temps_c. This has been done for you, so hit 'Submit Answer' to see the result!

# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')

# Print first 5 rows of temps_c
print(temps_c.head())

# Computing percentage growth of GDP
# Your job in this exercise is to compute the yearly percent-change of US GDP (Gross Domestic Product) since 2008.
#
# The data has been obtained from the Federal Reserve Bank of St. Louis and is available in the file GDP.csv, which contains quarterly data; you will resample it to annual sampling and then compute the annual growth of GDP. For a refresher on resampling, check out the relevant material from pandas Foundations.

# Read the file 'GDP.csv' into a DataFrame called gdp.
# Use parse_dates=True and index_col='DATE'.
# Create a DataFrame post2008 by slicing gdp such that it comprises all rows from 2008 onward.
# Print the last 8 rows of the slice post2008. This has been done for you. This data has quarterly frequency so the indices are separated by three-month intervals.
# Create the DataFrame yearly by resampling the slice post2008 by year. Remember, you need to chain .resample() (using the alias 'A' for annual frequency) with some kind of aggregation; you will use the aggregation method .last() to select the last element when resampling.
# Compute the percentage growth of the resampled DataFrame yearly with .pct_change() * 100

import pandas as pd

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv', index_col='DATE', parse_dates=True)

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

# Print yearly again
print(yearly)

# Converting currency of stocks
# In this exercise, stock prices in US Dollars for the S&P 500 in 2015 have been obtained from Yahoo Finance. The files sp500.csv for sp500 and exchange.csv for the exchange rates are both provided to you.
#
# Using the daily exchange rate to Pounds Sterling, your task is to convert both the Open and Close column prices.

# Read the DataFrames sp500 & exchange from the files 'sp500.csv' & 'exchange.csv' respectively..
# Use parse_dates=True and index_col='Date'.
# Extract the columns 'Open' & 'Close' from the DataFrame sp500 as a new DataFrame dollars and print the first 5 rows.
# Construct a new DataFrame pounds by converting US dollars to British pounds. You'll use the .multiply() method of dollars with exchange['GBP/USD'] and axis='rows'
# Print the first 5 rows of the new DataFrame pounds. This has been done for you, so hit 'Submit Answer' to see the results!.

# Import pandas
import pandas as pd

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', index_col='Date', parse_dates=True)

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', index_col='Date', parse_dates=True)

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis = 'rows')

# Print the head of pounds
print(pounds.head())

# append. Series and DataFrame method
s1.append(s2) # Stacks rows of s2 below s1

# concat is a pandas module function: accepts a list or seuqnce of several series of dfs to concatenate
pd.concat([s1, s2, s3]) # can stack row-wise or column-wise

# Using .reset_index()
northeast.append(south).reset_index(drop=True) # drops = True discards the old Index with repeated entries (rather than keeping it as a column in a DataFrame)

# use ignore_index
pd.concat([northeast, south], ignore_index=True)

# Appending Series with nonunique Indices
# The Series bronze and silver, which have been printed in the IPython Shell, represent the 5 countries that won the most bronze and silver Olympic medals respectively between 1896 & 2008. The Indexes of both Series are called Country and the values are the corresponding number of medals won.
#
# If you were to run the command combined = bronze.append(silver), how many rows would combined have? And how many rows would combined.loc['United States'] return? Find out for yourself by running these commands in the IPython Shell.

# Appending pandas Series
# In this exercise, you'll load sales data from the months January, February, and March into DataFrames. Then, you'll extract Series with the 'Units' column from each and append them together with method chaining using .append().
#
# To check that the stacking worked, you'll print slices from these Series, and finally, you'll add the result to figure out the total units sold in the first quarter.

# Read the files 'sales-jan-2015.csv', 'sales-feb-2015.csv' and 'sales-mar-2015.csv' into the DataFrames jan, feb, and mar respectively.
# Use parse_dates=True and index_col='Date'.
# Extract the 'Units' column of jan, feb, and mar to create the Series jan_units, feb_units, and mar_units respectively.
# Construct the Series quarter1 by appending feb_units to jan_units and then appending mar_units to the result. Use chained calls to the .append() method to do this.
# Verify that quarter1 has the individual Series stacked vertically. To do this:
# Print the slice containing rows from jan 27, 2015 to feb 2, 2015.
# Print the slice containing rows from feb 26, 2015 to mar 7, 2015.
# Compute and print the total number of units sold from the Series quarter1. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('sales-mar-2015.csv', index_col='Date', parse_dates=True)

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())

# Concatenating pandas Series along row axis
# Having learned how to append Series, you'll now learn how to achieve the same result by concatenating Series instead. You'll continue to work with the sales data you've seen previously. This time, the DataFrames jan, feb, and mar have been pre-loaded.
#
# Your job is to use pd.concat() with a list of Series to achieve the same result that you would get by chaining calls to .append().
#
# You may be wondering about the difference between pd.concat() and pandas' .append() method. One way to think of the difference is that .append() is a specific case of a concatenation, while pd.concat() gives you more flexibility, as you'll see in later exercises.

# Create an empty list called units. This has been done for you.
# Use a for loop to iterate over [jan, feb, mar]:
# In each iteration of the loop, append the 'Units' column of each DataFrame to units.
# Concatenate the Series contained in the list units into a longer Series called quarter1 using pd.concat().
# Specify the keyword argument axis='rows' to stack the Series vertically.
# Verify that quarter1 has the individual Series stacked vertically by printing slices. This has been done for you, so hit 'Submit Answer' to see the result!

# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])

# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis='rows')

# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])



