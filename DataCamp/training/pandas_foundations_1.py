import pandas as pd

import matplotlib.pyplot as plt

# Use dictionaries: Keys of the dictinary data are used as col labels
data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'visitors': [139, 237, 326, 456],
        'signups': [7, 12, 3, 5]
        }

users = pd.DataFrame(data)

print(users)

# DataFrames from dict(2)
# Build dataframes from Lists
# weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
# type(weekdays)

users['fees'] = 0 # Broadcasts to entire column

print(users)

heights = [59.0, 65.2, 62.9, 65.4, 63.7, 65.7, 64.1]

data = {'height': heights, 'sex': 'M'}

results = pd.DataFrame(data)

print(results)

# We can change the column and index labels using columns and index attributes of a Pandas DataFrame

results.columns = ['height (in)', 'sex']
results.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

print(results)

results.shape

# read_csv function requires a string describing a filepath
for filename in csv_files:
        data = pd.read_csv(filename)
        list_data.append(data)

def my_square(x):
    return x ** 2

df.apply(my_square)

df.apply(lambda x: x ** 2)

def square():
        new_value = 4 ** 2
        print(new_value)

# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout
shout()

# If you want to select every other value from the first 5 elements in the list
# left inclusive right exclusive nature of Python
l = [0, 1, 2, 3, 4]

l[0:5:2]

l[::2]


from math import factorial as fac
n = 5
k = 3

fac(n) / (fac(k) * fac(n-k))

if True:
    print("It's True")

