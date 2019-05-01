# Also here: https://github.com/IBMDeveloperUK/pandas-workshop
# Download measurements data from here: https://raw.githubusercontent.com/the-pudding/data/master/pockets/measurements.csv

import numpy as np
import pandas as pd

jeans = pd.read_csv('/Users/dimitrispolychronopoulos/Desktop/measurements.csv')

jeans.columns

dates = pd.date_range('20130101', periods=6)

numbers = np.random.randn(6, 4)
numbers

dates = pd.date_range('20130101', periods=6)
dates

df = pd.DataFrame(numbers, index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3] * 4, dtype='int32'),
                     'E': pd.Categorical(["test", "train", "test", "train"]),
                     'F': 'foo'})

print('Data type of s is ' + str(type(s)))

# It is important to understand the index to work with dataframes, so let's explore this a little further.
#
# For this we will create a new DataFrame with the population of the 5 largest cities in the UK (source). data is a dictionary.

data = {'city':       ['London','Manchester','Birmingham','Leeds','Glasgow'],
        'population': [9787426,  2553379,     2440986,    1777934,1209143],
        'area':       [1737.9,   630.3,       598.9,      487.8,  368.5 ]}

cities = pd.DataFrame(data)

cities.columns

cities.set_index('city')

cities = cities.reset_index()

cities = cities.set_index(['city','population'])

cities = cities.reset_index()
cities = cities.set_index('city')
cities

# Or select by position with .iloc[]. You can select a single row, multiple rows (or columns) at particular positions in the index, it only takes integers:

cities.loc['Manchester':'Leeds', ['area', 'population']]

cities.iloc[:,0:2]

cities.iloc[2:4,0:2]

cities2 = cities[['area','population']]
cities2

cities[(cities['area'] > 500) & (cities['population'] > 2500000)]

print(jeans['price'].max() - jeans['price'].min())

# Adding a column can be done by defining a new column, which can then be dropped with 'drop'.
jeans['new'] = 1

jeans = jeans.drop(columns='new'

jeans['avgHeightFront'] = (jeans.maxHeightFront + jeans.minHeightFront) / 2

# Things to check:
#
# Is the data tidy: each variable forms a column, each observation forms a row and each type of observational unit forms a table.
# Are all columns in the right data format?
# Are there missing values?
# Are there unrealistic outliers?

# Get a quick overview of the numeric data with `.describe()`. If any of the numeric columns is missing this is a probably because of a wring data type.

# It is not always ideal to have text in the table. Especially not if you want to create a model from the data. You could replace style into numbers, but is one style really twice as large as another. It is better to transform the data with get.dummies(). The below will add 4 new columns to the DataFrame:

jeans2 = jeans.copy()

style = pd.get_dummies(jeans2['style'], drop_first=True)

jeans2 = jeans2.join(style)

# Or do this all in one code:
style = pd.get_dummies(jeans2['style'], drop_first=True)

data = {'city':       ['London','Manchester','Birmingham','Leeds','Glasgow'],
        'population': [9787426,  2553379,     2440986,    1777934,1209143],
        'area':       [1737.9,   630.3,       598.9,      487.8,  368.5 ]}
cities = pd.DataFrame(data)

data2 = {'city':       ['Liverpool','Southampton'],
        'population': [864122,  855569],
        'area':       [199.6,   192.0]}
cities2 = pd.DataFrame(data2)

# These new cities can be added with append():
cities = cities.append(cities2)
cities

data = {'city': ['London','Manchester','Birmingham','Leeds','Glasgow'],
        'density': [5630,4051,4076,3645,3390]}
cities3 = pd.DataFrame(data)

# An extra column can be added with .merge() with an outer join using the city names:

cities = pd.merge(cities, cities3, how='outer', sort=True,on='city')

# Data that does not quite fit can be merged as well:

data = {'city':       ['Newcastle','Nottingham'],
        'population': [774891,  729977],
        'area':       [180.5,   176.4]}

cities4 = pd.DataFrame(data)

cities = cities.append(cities4)
cities

# Grouping data is a quick way to calculate values for classes in your DataFrame. The example below gives you the mean values of all variables for the 2 cutout classes, and for a comination of all classes when cutout and style are combined.

jeans.groupby(['cutout']).mean()

jeans.groupby(['cutout', 'style']).max().head(10)

# Add two columns `men` and `women` with `get_dummies()` and keep the original `menWomen`
jeans = jeans.join(pd.get_dummies(jeans['menWomen'], drop_first=False))

# Using `groupby().count()`: what is the number of mens and womens jeans measured?
jeans.groupby(['menWomen']).count()

# What are the average front and back pocket sizes of mens and womens jeans?
menfront=(jeans['FrontArea'][jeans['men'] == 1].mean())
# or jeans[jeans['menWomen'] == "men"].FrontArea.mean()
menback=(jeans['BackArea'][jeans['men'] == 1].mean())
womenfront=(jeans['FrontArea'][jeans['women'] == 1].mean())
womenback=(jeans['BackArea'][jeans['women'] == 1].mean())

print('Avg men front pocket size is ' + str(menfront))
print('Avg men back pocket size is ' + str(menback))
print('Avg women front pocket size is ' + str(womenfront))
print('Avg women back pocket size is ' + str(womenback))

# To find out how many unique values there are in a column use np.unique(df['a'])

# without this the plots would be opened  in a new window (not browser)
# with this instruction plots will be included in the notebook
# %matplotlib inline

import matplotlib.pyplot as plt

# The default plot is a line chart:
jeans['price'].plot();

# To create a plot that makes more sense for this data have a look at the documentation for all options. A histogram might work better. Go ahead and change the number of bins until you think the number of bins looks right:

jeans['price'].plot.hist(bins=5);

# Change the size of the plot with figsize:
jeans['price'].plot.hist(bins=15,figsize=(10,5));

jeans['price'][jeans['menWomen']=='men'].plot.hist(bins=15, figsize=(10,5))

# To add the womens jeans, simply repeat the plot command with a different selection of the data:
jeans['price'][jeans['menWomen']=='men'].plot.hist(bins=15,figsize=(10,5));
jeans['price'][jeans['menWomen']=='women'].plot.hist(bins=15,figsize=(10,5));

# The above plot is difficult to read as the histograms overlap. You can fix this by changing the colours and making them transparant. To add a legend each histogram needs to be assigned to an object ax that is used to create a legend:

ax = jeans['price'][jeans['menWomen']=='men'].plot.hist(
    bins=15,figsize=(10,5),alpha=0.5,color='#1A4D3B');

ax = jeans['price'][jeans['menWomen']=='women'].plot.hist(
    bins=15,figsize=(10,5),alpha=0.5,color='#4D1A39');

ax.legend(['men','women']);

# It is easy to change pretty much everything as in the below code. This was the ugliest I could come up with. Can you make it worse?
jeans['price'].plot.hist(
    bins=15,
    title="Jeans Price",
    legend=False,
    fontsize=14,
    grid=False,
    linestyle='--',
    edgecolor='black',
    color='darkred',
    linewidth=3);

# You can use groupby() in combination with a bar plot to visualize the price by style:
style = jeans['price'].groupby(jeans['style']).mean()
ax=style.plot.bar();
ax.set_ylabel('Jeans Price');

import seaborn as sns

sns.catplot(x='menWomen', y='price', data=jeans);

sns.catplot(x='menWomen', y='price', hue='style', kind='violin', data=jeans);

sns.catplot(x="style", y="price", kind="box", data=jeans);

sns.catplot(x="style", y="price", hue="menWomen", kind="box", data=jeans);

ax=sns.scatterplot(y='BackArea', x='price', data=jeans)
ax=sns.scatterplot(y='FrontArea', x='price', data=jeans)
ax.set_ylabel('Pocket Size');
ax.legend(['Back pocket','Front pocket']);

# Create two histograms that compare the sizes of pockets between men and womens jeans with `.plot.hist()`
ax = jeans['BackArea'][jeans['menWomen']=='men'].plot.hist(
    bins=15,figsize=(10,5),alpha=0.5);
ax = jeans['BackArea'][jeans['menWomen']=='women'].plot.hist(
    bins=15,figsize=(10,5),alpha=0.5);
ax.set_ylabel('Back Pocket Size');
ax.legend(['men','women']);

# Create a bar plot with the size of the front pockets for men and women with `.plot.bar()`
pockets = jeans.groupby('menWomen')['FrontArea'].mean()
ax = pockets.plot.bar();
ax.set_ylabel('Front Pocket Size');

# Create a bar plot with the size of the front pockets for men and women with `seaborn`
sns.barplot(x = "menWomen", y = 'FrontArea', data=jeans)
