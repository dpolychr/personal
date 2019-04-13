# Manipulating lists for fun and profit
# You may be familiar with adding individual data elements to a list by using the .append() method. However, if you want to combine a list with another array type (list, set, tuple), you can use the .extend() method on the list.
#
# You can also use the .index() method to find the position of an item in a list. You can then use that position to remove the item with the .pop() method.
#
# In this exercise, you'll practice using all these methods!
#
# Create a list called baby_names with the names 'Ximena', 'Aliza', 'Ayden', and 'Calvin'.
# Use the .extend() method on baby_names to add 'Rowen' and 'Sandeep' and print the list.
# Use the .index() method to find the position of 'Aliza' in the list. Save the result as position.
# Use the .pop() method with position to remove 'Aliza' from the list.
# Print the baby_names list. This has been done for you, so hit 'Submit Answer' to see the results!

# Create a list containing the names: baby_names
baby_names = ['Ximena', 'Aliza', 'Ayden', 'Calvin']

# Extend baby_names with 'Rowen' and 'Sandeep'
baby_names.extend(['Rowen', 'Sandeep'])

# Print baby_names
print(baby_names)

# Find the position of 'Aliza': position
position = baby_names.index('Aliza')

# Remove 'Aliza' from baby_names
baby_names.pop(position)

# Print baby_names
print(baby_names)

# Looping over lists
# You can use a for loop to iterate through all the items in a list. You can take that a step further with the sorted() function which will sort the data in a list from lowest to highest in the case of numbers and alphabetical order if the list contains strings.
#
# The sorted() function returns a new list and does not affect the list you passed into the function. You can learn more about sorted() in the Python documentation.
#
# A list of lists, records has been pre-loaded. If you explore it in the IPython Shell, you'll see that each entry is a list of this form:
#
# ['2011', 'FEMALE', 'HISPANIC', 'GERALDINE', '13', '75']
#
# The name of the baby ('GERALDINE') is the fourth entry of this list. Your job in this exercise is to loop over this list of lists and append the names of each baby to a new list called baby_names.
#
# Create an empty list called baby_names.
# Use a for loop to iterate over each row of records:
# Append the name in records to baby_names. The name is stored in the fourth element of row.
# Print each name in baby_names in alphabetical order. To do this:
# Use the sorted() function as part of a for loop to iterate over the sorted names, printing each one.

# Create the empty list: baby_names
baby_names = []

# Loop over records
for row in records:
    # Add the name to the list
    baby_names.append(row[3])

# Sort the names in alphabetical order
for name in sorted(baby_names):
    # Print each name
    print(name)

# Tuple as a container type
#
# Tuples hold data in order
#
# Index
#
# Tuples are easier to process and more memory efficient than lists
#
# Tuples are immutable which means we cannot add or remove elements from them
#
# Tuples are commonly created by zipping lists together with zip()
#
# Using and unpacking tuples
# Tuples are made of several items just like a list, but they cannot be modified in any way. It is very common for tuples to be used to represent data from a database. If you have a tuple like ('chocolate chip cookies', 15) and you want to access each part of the data, you can use an index just like a list. However, you can also "unpack" the tuple into multiple variables such as type, count = ('chocolate chip cookies', 15) that will set type to 'chocolate chip cookies' and count to 15.
#
# Often you'll want to pair up multiple array data types. The zip() function does just that. It will return a list of tuples containing one element from each list passed into zip().
#
# When looping over a list, you can also track your position in the list by using the enumerate() function. The function returns the index of the list item you are currently on in the list and the list item itself.
#
# You'll practice using the enumerate() and zip() functions in this exercise, in which your job is to pair up the most common boy and girl names. Two lists - girl_names and boy_names - have been pre-loaded into your workspace.
#
# Use the zip() function to pair up girl_names and boy_names into a variable called pairs.
# Use a for loop to loop through pairs, using enumerate() to keep track of your position. Unpack pairs into the variables idx and pair.
# Inside the for loop:
# Unpack pair into the variables girl_name and boy_name.
# Print the rank, girl name, and boy name, in that order. The rank is contained in idx.

# Pair up the boy and girl names: pairs
pairs = zip(girl_names, boy_names)

# Iterate over pairs
for idx, pair in enumerate(pairs):
    # Unpack pair: girl_name, boy_name
    girl_name, boy_name = pair
    # Print the rank and names associated with each rank
    print('Rank {}: {} and {}'.format(idx, girl_name, boy_name))

# Making tuples by accident
# Tuples are very powerful and useful, and it's super easy to make one by accident. All you have to do is create a variable and follow the assignment with a comma. This becomes an error when you try to use the variable later expecting it to be a string or a number.
#
# You can verify the data type of a variable with the type() function. In this exercise, you'll see for yourself how easy it is to make a tuple by accident.

# Create the normal variable: normal
normal = 'simple'

# Create the mistaken variable: error
error = 'trailing comma',

# Print the types of the variables
print(type(normal))
print(type(error))

# Sets are created from a list
cookies_eaten_today = ['chocolate chip', 'peanut butter', 'chocolate chip', 'oatmeal cream', 'chocolate chip']

types_cookies_eaten = set(cookies_eaten_today)

types_cookies_eaten.add('biscotti')

types_cookies_eaten.add('chocolate chip')

cookies_hugo_ate = ['chocolate chip', 'anzac']

types_cookies_eaten.update(cookies_hugo_ate)

print(types_cookies_eaten)

# .discard safely removes an element from the set by value

# Finding all the data and the overlapping data between sets
# Sets have several methods to combine, compare, and study them all based on mathematical set theory. The .union() method returns a set of all the names found in the set you used the method on plus any sets passed as arguments to the method. You can also look for overlapping data in sets by using the .intersection() method on a set and passing another set as an argument. It will return an empty set if nothing matches.
#
# Your job in this exercise is to find the union and intersection in the names from 2011 and 2014. For this purpose, two sets have been pre-loaded into your workspace: baby_names_2011 and baby_names_2014.
#
# One quirk in the baby names dataset is that names in 2011 and 2012 are all in upper case, while names in 2013 and 2014 are in title case (where the first letter of each name is capitalized). Consequently, if you were to compare the 2011 and 2014 data in this form, you would find no overlapping names between the two years! To remedy this, we converted the names in 2011 to title case using Python's .title() method.
#
# Real-world data can often come with quirks like this - it's important to catch them to ensure your results are meaningful.

# Combine all the names in baby_names_2011 and baby_names_2014 by computing their union. Store the result as all_names.
# Print the number of names that occur in all_names. You can use the len() function to compute the number of names in all_names.
# Find all the names that occur in both baby_names_2011 and baby_names_2014 by computing their intersection. Store the result as overlapping_names.
# Print the number of names that occur in overlapping_names.

# Find the union: all_names
all_names = baby_names_2011.union(baby_names_2014)

# Print the count of names in all_names
print(len(all_names))

# Find the intersection: overlapping_names
overlapping_names = baby_names_2011.intersection(baby_names_2014)

# Print the count of names in overlapping_names
print(len(overlapping_names))

# Determining set differences
# Another way of comparing sets is to use the difference() method. It returns all the items found in one set but not another. It's important to remember the set you call the method on will be the one from which the items are returned. Unlike tuples, you can add() items to a set. A set will only add items that do not exist in the set.
#
# In this exercise, you'll explore what names were common in 2011, but are no longer common in 2014. The set baby_names_2014 has been pre-loaded into your workspace. As in the previous exercise, the names have been converted to title case to ensure a proper comparison.
#
# Instructions
# 100 XP
# Create an empty set called baby_names_2011. You can do this using set().
# Use a for loop to iterate over each row in records:
# If the first column of each row in records is '2011', add its fourth column to baby_names_2011. Remember that Python is 0-indexed!
# Find the difference between baby_names_2011 and baby_names_2014. Store the result as differences.
# Print the differences. This has been done for you, so hit 'Submit Answer' to see the result!

# Create the empty set: baby_names_2011
baby_names_2011 = set()

# Loop over records and add the names from 2011 to the baby_names_2011 set
for row in records:
    # Check if the first column is '2011'
    if row[0] == '2011':
        # Add the fourth column to the set
        baby_names_2011.add(row[3])

# Find the difference between 2011 and 2014: differences
differences = baby_names_2011.difference(baby_names_2014)

# Print the differences
print(differences)

# Safely finding by key

# Creating and looping through dictionaries
# You'll often encounter the need to loop over some array type data, like in Chapter 1, and provide it some structure so you can find the data you desire quickly.
#
# You start that by creating an empty dictionary and assigning part of your array data as the key and the rest as the value.
#
# Previously, you used sorted() to organize your data in a list. Dictionaries can also be sorted. By default, using sorted() on a dictionary will sort by the keys of the dictionary. You can also reverse the order by passing reverse=True as a keyword argument.
#
# Finally, since sorted returns a list, you can use slice notation to select only part of the list. For example, [:10] will slice the first ten items off a list and return only those items.

# Create an empty dictionary called names.
# Loop over female_baby_names_2012, unpacking it into the variables name and rank.
# Inside the loop, add each name to the names dictionary using the rank as the key.
# Sort the names dictionary in descending order, select the first ten items. Print each item.

# Create an empty dictionary: names
names = dict()

# Loop over the girl names
for name, rank in female_baby_names_2012:
    # Add each name to the names dictionary using rank as the key
    names[rank] = name

# Sort the names list by rank in descending order and slice the first 10 items
for rank in sorted(names, reverse=True)[:10]:
    # Print each item
    print(names[rank])

# Safely finding by key
# As demonstrated in the video, if you attempt to access a key that isn't present in a dictionary, you'll get a KeyError. One option to handle this type of error is to use a try: except: block. You can learn more about error handling in Python Data Science Toolbox (Part 1), specifically in this video.
#
# Python provides a faster, more versatile tool to help with this problem in the form of the .get() method. The .get() method allows you to supply the name of a key, and optionally, what you'd like to have returned if the key is not found.
#
# You'll be using same names dictionary from the previous exercise and will gain practice using the .get() method.

# Safely print rank 7 from the names dictionary.
# Safely print the type of rank 100 from the names dictionary.
# Safely print rank 105 from the names dictionary or 'Not Found' if 105 is not found.

# Safely print rank 7 from the names dictionary
print(names.get(7))

# Safely print the type of rank 100 from the names dictionary
print(type(names.get(100)))

# Safely print rank 105 from the names dictionary or 'Not Found'
print(names.get(105, 'Not Found'))

# Dealing with nested data
# A dictionary can contain another dictionary as the value of a key, and this is a very common way to deal with repeating data structures such as yearly, monthly or weekly data. All the same rules apply when creating or accessing the dictionary.
#
# For example, if you had a dictionary that had a ranking of my cookie consumption by year and type of cookie. It might look like cookies = {'2017': {'chocolate chip': 483, 'peanut butter': 115}, '2016': {'chocolate chip': 9513, 'peanut butter': 6792}}. I could access how many chocolate chip cookies I ate in 2016 using cookies['2016']['chocolate chip'].
#
# When exploring a new dictionary, it can be helpful to use the .keys() method to get an idea of what data might be available within the dictionary. You can also iterate over a dictionary and it will return each key in the dictionary for you to use inside the loop. Here, a dictionary called boy_names has been loaded into your workspace. It consists of all male names in 2013 and 2014.

# Print the keys of the boy_names dictionary.
# Print the keys of the boy_names dictionary for the year 2013.
# Loop over the boy_names dictionary.
# Inside the loop, safely print the year and the third ranked name. Print 'Unknown' if the third ranked name is not found.

# Print a list of keys from the boy_names dictionary
print(boy_names.keys())

# Print a list of keys from the boy_names dictionary for the year 2013
print(boy_names[2013].keys())

# Loop over the dictionary
for year in boy_names:
    # Safely print the year and the third ranked name or 'Unknown'
    print(year, boy_names[year].get(2, 'Unknown'))

# Dicts are mutable so we can alter them in a number of ways
# Adding and extending dicts

# update method to update a dic from another dic, tuples or keywords
# Adding and extending dictionaries
# If you have a dictionary and you want to add data to it, you can simply create a new key and assign the data you desire to it. It's important to remember that if it's a nested dictionary, then all the keys in the data path must exist, and each key in the path must be assigned individually.
#
# You can also use the .update() method to update a dictionary with keys and values from another dictionary, tuples or keyword arguments.
#
# Here, you'll combine several techniques used in prior exercises to setup your dictionary in a way that makes it easy to find the least popular baby name for each year.
#
# Your job is to add data for the year 2011 to your dictionary by assignment, 2012 by update, and then find the least popular baby name for each year.

# Assign the names_2011 dictionary as the value to the 2011 key of the boy_names dictionary.
# Update the 2012 key in the boy_names dictionary with the following data in a list of tuples: (1, 'Casey'), (2, 'Aiden').
# Loop over the boy_names dictionary.
# Inside the first for loop, use another for loop to loop over and sort the data for each year of boy_names by descending rank.
# Make sure you have a rank and print 'No Data Available' if not. This has been done for you.
# Safely print the year and least popular name or 'Not Available' if it is not found. Take advantage of the .get() method.

# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011

# Update the 2012 key in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])

# Loop over the boy_names dictionary
for year in boy_names:
    # Loop over and sort the data for each year by descending rank
    for rank in sorted(boy_names[year], reverse=True)[:1]:
        # Check that you have a rank
        if not rank:
            print(year, 'No Data Available')
        # Safely print the year and the least popular name or 'Not Available'
        print(year, boy_names[year].get(rank, 'Not Available'))

# Popping and deleting from dictionaries
# Often, you will want to remove keys and value from a dictionary. You can do so using the del Python instruction. It's important to remember that del will throw a KeyError if the key you are trying to delete does not exist. You can not use it with the .get() method to safely delete items; however, it can be used with try: catch:.
#
# If you want to save that deleted data into another variable for further processing, the .pop() dictionary method will do just that. You can supply a default value for .pop() much like you did for .get() to safely deal with missing keys. It's also typical to use .pop() instead of del since it is a safe method.
#
# Here, you'll remove 2011 and 2015 to save them for later, and then delete 2012 from the dictionary.

# Remove 2011 from female_names and store it as female_names_2011.
# Safely remove 2015 from female_names with a empty dictionary as the default and store it as female_names_2015. To do this, pass in an empty dictionary {} as a second argument to .pop().
# Delete 2012 from female_names.
# Print female_names.

# Remove 2011 and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015, {})

# Delete 2012
del female_names[2012]

# Print female_names
print(female_names)

# Working with dictionaries
# .items() method returns an object we can iterate over

# Checking dictionaries for data
# .get() does a lot of work to check for a key
# in operator is much more efficient and clearer

# Working with dictionaries more pythonically
# So far, you've worked a lot with the keys of a dictionary to access data, but in Python, the prefered manner for iterating over items in a dictionary is with the .items() method.

# This returns each key and value from the dictionary as a tuple, which you can unpack in a for loop. You'll now get practice doing this.

# Iterate over the 2014 nested dictionary
for rank, name in baby_names[2014].items():
    # Print rank and name
    print(rank, name)

# Iterate over the 2012 nested dictionary
for rank, name in baby_names[2012].items():
    # Print rank and name
    print(rank, name)

# Checking dictionaries for data
# You can check to see if a key exists in a dictionary by using the in expression.
#
# For example, you can check to see if 'cookies' is a key in the dictionary by using if 'cookies' in recipes_dict: this allows you to safely react to data being present in the dictionary.
#
# You can also use the in expression so see if data is in the value of a dictionary such as if 'cookies' in recipes_dict.values(). Remember you have to handle nested dictionaries differently as illustrated in the video and previous exercises, and use the in expression on each nested dictionary.

# Check to see if 2011 is in the baby_names dictionary.
# Print 'Found 2011' if it is present.
# Check to see if 1 is in baby_names[2012].
# Print 'Found Rank 1 in 2012' if found and 'Rank 1 missing from 2012' if not found.
# Check to see if rank 5 is in baby_names[2013].
# Print 'Found Rank 5' if it is present.

# Check to see if 2011 is in baby_names
if '2011' in baby_names:
    # Print 'Found 2011'
    print('Found 2011')

# Check to see if rank 1 is in 2012
if 1 in baby_names[2012]:
    # Print 'Found Rank 1 in 2012' if found
    print('Found Rank 1 in 2012')
else:
    # Print 'Rank 1 missing from 2012' if not found
    print('Rank 1 missing from 2012')

# Check to see if Rank 5 is in 2013
if 5 in baby_names[2013]:
    # Print 'Found Rank 5'
    print('Found Rank 5')

import csv
import os
os.getcwd()
csvfile = open('/Users/dimitrispolychronopoulos/Downloads/RDA_240_summary.csv', 'r')
for row in csv.reader(csvfile):
    print(row)

csvfile.close()

# Creating a dictionary from a file using DictReader
for row in csv.DictReader(csvfile):
    print(row)

# Reading from a file using CSV reader
# Python provides a wonderful module called csv to work with CSV files. You can pass the .reader() method of csv a Python file object and use it as you would any other iterable. To create a Python file object, you use the open() function, which accepts a file name and a mode. The mode is typically 'r' for read or 'w' for write.
#
# Though you won't use it for this exercise, often CSV files will have a header row with field names, and you will need to use slice notation such as [1:] to skip the header row.
#
# You'll now use the csv module to read the baby_names.csv file and fill the baby_names dictionary with data. This baby_names dictionary has already been created for you.

# Import the python csv module.
# Create a Python file object in read mode for baby_names.csv called csvfile.
# Loop over a csv reader on the file object. Inside the loop:
# Print each row.
# Add the rank (the 6th element of row) as the key and name (the 4th element of row) as the value to the existing dictionary (baby_names).
# Print the keys of baby_names.

# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row
    print(row)
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())

# Creating a dictionary from a file
# The csv module also provides a way to directly create a dictionary from a CSV file with the DictReader class. If the file has a header row, that row will automatically be used as the keys for the dictionary. However, if not, you can supply a list of keys to be used. Each row from the file is returned as a dictionary. Using DictReader can make it much easier to read your code and understand what data is being used, especially when compared to the numbered indexes you used in the prior exercise.
#
# Your job in this exercise is to create a dictionary directly from the data file using DictReader. NOTE: The misspellings are from the original data, and this is a very common issue. Again, the baby_names dictionary has already been created for you

# Import the Python csv module.
# Create a Python file object in read mode for the baby_names.csv called csvfile.
# Loop over a csv DictReader on csvfile. Inside the loop:
# Print each row.
# Add the 'RANK' of each row as the key and 'NAME' of each row as the value to the existing dictionary.
# Print the dictionary keys.

# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())

# Counter: special dictionary used for counting data, measuring frequency

# Using Counter on lists
# Counter is a powerful tool for counting, validating, and learning more about the elements within a dataset that is found in the collections module. You pass an iterable (list, set, tuple) or a dictionary to the Counter. You can also use the Counter object similarly to a dictionary with key/value assignment, for example counter[key] = value.
#
# A common usage for Counter is checking data for consistency prior to using it, so let's do just that. In this exercise, you'll be using data from the Chicago Transit Authority on ridership.

# Import the Counter object from collections.
# Print the first ten items from the stations list.
# Create a Counter of the stations list called station_count.
# Print the station_count.

# Import the Counter object
from collections import Counter

# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Print the station_count
print(station_count)

# Finding most common elements
# Another powerful usage of Counter is finding the most common elements in a list. This can be done with the .most_common() method.
#
# Practice using this now to find the most common stations in a stations list.

# Import the Counter object from collections.
# Create a Counter of the stations list called station_count.
# Print the 5 most common elements.

# Import the Counter object
from collections import Counter

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Find the 5 most common elements
print(station_count.most_common(5))

# Defaultdict

# Pass it a default type that every key will have even if it does not currently exist
# Works exactly like a dic

# Creating dictionaries of an unknown structure
# Occasionally, you'll need a structure to hold nested data, and you may not be certain that the keys will all actually exist. This can be an issue if you're trying to append items to a list for that key. You might remember the NYC data that we explored in the video. In order to solve the problem with a regular dictionary, you'll need to test that the key exists in the dictionary, and if not, add it with an empty list.
#
# You'll be working with a list of entries that contains ridership details on the Chicago transit system. You're going to solve this same type of problem with a much easier solution in the next exercise.

# Create an empty dictionary called ridership.
# Iterate over entries, unpacking it into the variables date, stop, and riders.
# Check to see if the date already exists in the ridership dictionary. If it does not exist, create an empty list for the date key.
# Append a tuple consisting of stop and riders to the date key of the ridership dictionary.
# Print the ridership for '03/09/2016'.

# Create an empty dictionary: ridership
ridership = dict()

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the dictionary
    if date not in ridership:
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))

# Print the ridership for '03/09/2016'
print(ridership['03/09/2016'])

# Safely appending to a key's value list
# Often when working with dictionaries, you know the data type you want to have each key be; however, some data types such as lists have to be initialized on each key before you can append to that list.
#
# A defaultdict allows you to define what each uninitialized key will contain. When establishing a defaultdict, you pass it the type you want it to be, such as a list, tuple, set, int, string, dictionary or any other valid type object.

# Import defaultdict from collections.
# Create a defaultdict with a default type of list called ridership.
# Iterate over the list entries, unpacking it into the variables date, stop, and riders, exactly as you did in the previous exercise.
# Use stop as the key of the ridership dictionary and append riders to its value.
# Print the first 10 items of the ridership dictionary. You can use the .items() method for this. Remember, you have to convert ridership.items() to a list before slicing.

# Import defaultdict
from collections import defaultdict

# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    ridership[stop].append(riders)

# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:10])

# Order in Python dicts
# Python version < 3.6 NOT ordered

from collections import OrderedDict

# .popitem() method returns items in reverse insertion order

# Working with OrderedDictionaries
# # Recently in Python 3.6, dictionaries were made to maintain the order in which the keys were inserted; however, in all versions prior to that you need to use an OrderedDict to maintain insertion order.
# #
# # Let's create a dictionary of all the stop times by route and rider, then use it to find the ridership throughout the day.

# Import OrderedDict from collections.
# Create an OrderedDict called ridership_date.
# Iterate over the list entries, unpacking it into date and riders.
# If a key does not exist in ridership_date for the date, set it equal to 0 (if only you could use defaultdict here!)
# Add riders to the date key of ridership_date.
# Print the first 31 records. Remember to convert the items into a list.

# Import OrderedDict from collections
from collections import OrderedDict

# Create an OrderedDict called: ridership_date
ridership_date = OrderedDict()

# Iterate over the entries
for date, riders in entries:
    # If a key does not exist in ridership_date, set it to 0
    if not date in ridership_date:
        ridership_date[date] = 0

    # Add riders to the date key in ridership_date
    ridership_date[date] += riders

# Print the first 31 records
print(list(ridership_date.items())[:31])

# Powerful Ordered popping
# Where OrderedDicts really shine is when you need to access the data in the dictionary in the order you added it. OrderedDict has a .popitem() method that will return items in reverse of which they were inserted. You can also pass .popitem() the last=False keyword argument and go through the items in the order of how they were added.
#
# Here, you'll use the ridership_date OrderedDict you created in the previous exercise.

# Print the first key in ridership_date (Remember to make keys a list before slicing).
# Pop the first item from ridership_date and print it.
# Print the last key in ridership_date.
# Pop the last item from ridership_date and print it.

# Print the first key in ridership_date
print(list(ridership_date.keys())[0])

# Pop the first item from ridership_date and print it
print(ridership_date.popitem(last=False))

# Print the last key in ridership_date
print(list(ridership_date.keys())[-1])

# Pop the last item from ridership_date and print it
print(ridership_date.popitem())

# namedtuple is a tuple where each position (column) has a name

# Creating namedtuples for storing data
# # Often times when working with data, you will use a dictionary just so you can use key names to make reading the code and accessing the data easier to understand. Python has another container called a namedtuple that is a tuple, but has names for each position of the tuple. You create one by passing a name for the tuple type and a list of field names.
# #
# # For example, Cookie = namedtuple("Cookie", ['name', 'quantity']) will create a container, and you can create new ones of the type using Cookie('chocolate chip', 1) where you can access the name using the name attribute, and then get the quantity using the quantity attribute.
# #
# # In this exercise, you're going to restructure the transit data you've been working with into namedtuples for more descriptive code.

# Import namedtuple from collections.
# Create a namedtuple called DateDetails with a type name of DateDetails and fields of 'date', 'stop', and 'riders'.
# Create a list called labeled_entries.
# Iterate over entries, unpacking it into date, stop, and riders.
# Create a new DateDetails namedtuple instance for each entry and append it to labeled_entries.
# Print the first 5 items in labeled_entries. This has been done for you, so hit 'Submit Answer' to see the result!

# Import namedtuple from collections
from collections import namedtuple

# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))

# Print the first 5 items in labeled_entries
print(labeled_entries[:5])

# Leveraging attributes on namedtuples
# Once you have a namedtuple, you can write more expressive code that is easier to understand. Remember, you can access the elements in the tuple by their name as an attribute. For example, you can access the date of the namedtuples in the previous exercise using the .date attribute.
#
# Here, you'll use the tuples you made in the previous exercise to see how this works.

# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop
    print(item.stop)

    # Print each item's date
    print(item.date)

    # Print each item's riders
    print(item.riders)

from urllib.request import urlopen

with urlopen('http://sixty-north.com/c/t.txt') as story:
    story_words = []
    for line in story:
        line_words = line.decode('utf-8').split()
        for word in line_words:
            story_words.append(word)

for word in story_words:
    print(word)

# Strings to DateTimes
# Time to begin your DateTime journey! You'll start by using the .strptime() method from the datetime object as shown in the video, passing it both the string and the format. A full list of the format string components is available in the Python documentation.
#
# You'll be using the datetime column from the Chicago Transist Authority data, which is available as dates_list. Feel free to explore it in the IPython Shell: You'll see that it has the format of Month, Day, Year.

# Import the datetime object from datetime.
# Iterate over the dates_list, using date_str as your iterator variable.
# Convert each date_str into a datetime object called date_dt using the datetime.strptime() function, with '%m/%d/%Y' as your format.
# Print each date_dt.

# Import the datetime object from datetime
from datetime import datetime

# Iterate over the dates_list
for date_str in dates_list:
    # Convert each date to a datetime object: date_dt
    date_dt = datetime.strptime(date_str, '%m/%d/%Y')

    # Print each date_dt
    print(date_dt)

# Converting to a String
# Converting from a datetime object to a string is done with the .strftime() method on a instance of the datetime object. You pass a format string just like the ones used in the prior exercise.
#
# There is also a widely used string output standard called ISO-8601. It has a shortcut method named .isoformat(). I encourage you to use it anytime you write out to a file.
#
# All the datetimes you created for the transit data in the prior exercise are saved in the datetimes_list.

# Loop over the first 10 items of the datetimes_list, using item as your iterator variable.
# Print out the item as a string in the format of 'MM/DD/YYYY'. For this, the format string is '%m/%d/%Y'.
# Print out the item as an ISO standard string.

# Loop over the first 10 items of the datetimes_list
for item in datetimes_list[:10]:
    # Print out the record as a string in the format of 'MM/DD/YYYY'
    print(datetime.strftime(item, '%m/%d/%Y'))

    # Print out the record as an ISO standard string
    print(item.isoformat())

# Pieces of Time
# When working with datetime objects, you'll often want to group them by some component of the datetime such as the month, year, day, etc. Each of these are available as attributes on an instance of a datetime object.
#
# You're going to work with the summary of the CTA's daily ridership. It contains the following columns, in order: service_date, day_type, bus, rail_boardings, and total_rides. The modules defaultdict and datetime have already been imported for you.

# Create a defaultdict of an integer called monthly_total_rides.
# Loop over the list daily_summaries, which contains the columns mentioned above in the assignment text.
# Convert the service_date (1st element of daily_summary) to a datetime object called service_datetime. Use '%m/%d/%Y' as your format string.
# Use the month of the service_datetime as the dict key and add the total_rides (5th element of daily_summary) to the current amount for the month. Be sure to convert this into an integer.
# Print monthly_total_rides.

# Create a defaultdict of an integer: monthly_total_rides
monthly_total_rides = defaultdict(int)

# Loop over the list daily_summaries
for daily_summary in daily_summaries:
    # Convert the service_date to a datetime object
    service_datetime = datetime.strptime(daily_summary[0], '%m/%d/%Y')

    # Add the total rides to the current amount for the month
    monthly_total_rides[service_datetime.month] += int(daily_summary[4])

# Print monthly_total_rides
print(monthly_total_rides)

# Creating DateTime Objects... Now
# Often when working with datetime objects, you'll want to work on windows or ranges that start from the current date and time. You can do this using the datetime now functions. There is a .now() method on the datetime object in the datetime module and a .utcnow() method. The .now() method returns the current local time on the machine on which it is run, and .utcnow() does the same thing but returns the value in UTC time. You'll need to be very familiar with these methods.
#
# No dataset is used in this exercise, but bear with us as you'll need to do this often to compare year/month-to-date etc.

from datetime import datetime

# Import datetime from the datetime module
from datetime import datetime

# Compute the local datetime: local_dt
local_dt = datetime.now()

# Print the local datetime
print(local_dt)

# Compute the UTC datetime: utc_dt
utc_dt = datetime.utcnow()

# Print the UTC datetime
print(utc_dt)

# Timezones
# In order to work effectively with other timezones, you can use the pytz library. To use timezones, you need to import the timezone object from the pytz module. Then you can use the timezone constructor and pass it a name of a timezone, such as CT = timezone('US/Central'). You can get a full list of timezone names at Wikipedia. In Python 3, you can make a datetime object "aware" by passing a timezone as the tzinfo keyword argument to the .replace() method on a datetime instance.
#
# An "aware" datetime object has an .astimezone() method that accepts a timezone object and returns a new datetime object in the desired timezone. If the tzinfo is not set for the datetime object it assumes the timezone of the computer you are working on.
#
# A list, daily_summaries, has been supplied for you it contains the datetime and rail ridership for trains going to New York. You need to determine the time in New York so you can align it with the New York Transit Authority data.

# Create a Timezone object for Chicago ('US/Central') called chicago_usa_tz.
# Create a Timezone object for New York ('US/Eastern') called ny_usa_tz.
# Iterate over the daily_summaries, unpacking it into the variables orig_dt and ridership.
# Make the orig_dt timezone "aware" for Chicago, using chicago_usa_tz. Store the result in chicago_dt.
# Convert chicago_dt to the New York timezone, ny_dt.
# Print the chicago_dt, ny_dt, and ridership

# Create a Timezone object for Chicago
chicago_usa_tz = timezone('US/Central')

# Create a Timezone object for New York
ny_usa_tz = timezone('US/Eastern')

# Iterate over the daily_summaries list
for orig_dt, ridership in daily_summaries:
    # Make the orig_dt timezone "aware" for Chicago
    chicago_dt = orig_dt.replace(tzinfo=chicago_usa_tz)

    # Convert chicago_dt to the New York Timezone
    ny_dt = chicago_dt.astimezone(ny_usa_tz)

    # Print the chicago_dt, ny_dt, and ridership
    print('Chicago: %s, NY: %s, Ridership: %s' % (chicago_dt, ny_dt, ridership))

# Finding a time in the future and from the past
# Another common case when working with times is to get a date 30, 60, or 90 days in the past from some date. In Python, the timedelta object from the datetime module is used to represent differences in datetime objects. You can create a timedelta by passing any number of keyword arguments such as days, seconds, microseconds, milliseconds, minutes, hours, and weeks to timedelta().
#
# Once you have a timedelta object, you can add or subtract it from a datetime object to get a datetime object relative to the original datetime object.
#
# A dictionary, daily_summaries, has been supplied for you. It contains the datetime as the key with a dict as the value that has 'day_type' and 'total_ridership' keys. A list of datetimes to review called review_dates is also available.

# Import timedelta from the datetime module.
# Build a timedelta of 30 days called glanceback using timedelta().
# Iterate over the review_dates, using date as your iterator variable.
# Calculate the date 30 days back by subtracting glanceback from date.
# Print the date, along with 'day_type' and 'total_ridership' from daily_summaries for that date.
# Print the prior_period_dt, along with 'day_type' and 'total_ridership' from daily_summaries for that date (prior_period_dt).

# Import timedelta from the datetime module
from datetime import timedelta

# Build a timedelta of 30 days: glanceback
glanceback = timedelta(days=30)

# Iterate over the review_dates as date
for date in review_dates:
    # Calculate the date 30 days back: prior_period_dt
    prior_period_dt = date - glanceback

    # Print the review_date, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
          (date,
           daily_summaries[date]['day_type'],
           daily_summaries[date]['total_ridership']))

    # Print the prior_period_dt, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
          (prior_period_dt,
           daily_summaries[prior_period_dt]['day_type'],
           daily_summaries[prior_period_dt]['total_ridership']))

# Finding differences in DateTimes
# Just like you were able to subtract a timedelta from a datetime to find a date in the past, you can also calculate the difference between two dates to get the timedelta between in return. Here, you'll find out how much time has elapsed between two transit dates.
#
# A list of tuples called date_ranges is provided for you. We took the dates from our dataset at every 30th record, and we paired up the records into tuples in a stepwise fashion.

# Iterate over date_ranges, unpacking it into start_date and end_date.
# Print the end_date and start_date using the same print() function.
# Print the difference between each end_date and start_date

# Iterate over the date_ranges
for start_date, end_date in date_ranges:
    # Print the End and Start Date
    print(end_date, start_date)
    # Print the difference between each end and start date
    print(end_date - start_date)

# Localizing time with pendulum
# Here, you're going to use pendulum to practice doing some common datetime operations!

# Import the pendulum module.
# Create a now datetime for Tokyo ('Asia/Tokyo') called tokyo_dt.
# Covert tokyo_dt to Los Angeles time ('America/Los_Angeles'). Store the result as la_dt.
# Print the ISO 8601 string of la_dt, using the .to_iso8601_string() method.

# Import the pendulum module
import pendulum

# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')

# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')

# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())

# Humanizing Differences with Pendulum
# Pendulum provides a powerful way to convert strings to pendulum datetime objects via the .parse() method. Just pass it a date string and it will attempt to convert into a valid pendulum datetime. By default, .parse() can process dates in ISO 8601 format. To allow it to parse other date formats, pass strict = False.
#
# It also has a wonderful alternative to timedelta. When calculating the difference between two dates by subtraction, pendulum provides methods such as .in_days() to output the difference in a chosen metric. These are just the beginning of what pendulum can do for you.
#
# A list of tuples called date_ranges is provided for you. This is the same list of tuples that contain two dates that was used a few exercises prior. You'll be focusing on comparing ranges of records.
#
# You can learn more in the pendulum documentation. Here, it has been imported for you.

# Iterate over the date_ranges list, unpacking it into start_date and end_date. These dates are not in ISO 8601 format.
# Use pendulum to convert the start_date string to a pendulum date called start_dt.
# Use pendulum to convert the end_date string to pendulum date called end_dt.
# Calculate the difference between end_dt and start_dt. Store the result as diff_period.
# Print the difference in days, using the .in_days() method.

# Iterate over date_ranges
for start_date, end_date in date_ranges:
    # Convert the start_date string to a pendulum date: start_dt
    start_dt = pendulum.parse(start_date, strict=False)

    # Convert the end_date string to a pendulum date: end_dt
    end_dt = pendulum.parse(end_date, strict=False)

    # Print the End and Start Date
    print(end_dt, start_dt)

    # Calculate the difference between end_dt and start_dt: diff_period
    diff_period = end_dt - start_dt

    # Print the difference in days
    print(diff_period.in_days())

# Create and use a Counter with a slight twist
from collections import Counter
nyc_eatery_count_by_types = Counter(nyC_eatery_types)

# Reading your data with CSV Reader and Establishing your Data Containers
# Let's get started! The exercises in this chapter are intentionally more challenging, to give you a chance to really solidify your knowledge. Don't lose heart if you find yourself stuck; think back to the concepts you've learned in previous chapters and how you can apply them to this crime dataset. Good luck!
#
# Your data file, crime_sampler.csv contains the date (1st column), block where it occurred (2nd column), primary type of the crime (3rd), description of the crime (4th), description of the location (5th), if an arrest was made (6th), was it a domestic case (7th), and city district (8th).
#
# Here, however, you'll focus only 4 columns: The date, type of crime, location, and whether or not the crime resulted in an arrest.
#
# Your job in this exercise is to use a CSV Reader to load up a list to hold the data you're going to analyze.

# Import the Python csv module.
# Create a Python file object in read mode for crime_sampler.csv called csvfile.
# Create an empty list called crime_data.
# Loop over a csv reader on the file object :
# Inside the loop, append the date (first element), type of crime (third element), location description (fifth element), and arrest (sixth element) to the crime_data list.
# Remove the first element (headers) from the crime_data list.
# Print the first 10 records of the crime_data list. This has been done for you, so hit 'Submit Answer' to see the result!

# Import the csv module
import csv

# Create the file object: csvfile
csvfile = open('crime_sampler.csv', 'r')

# Create an empty list: crime_data
crime_data = []

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Append the date, type of crime, location description, and arrest
    crime_data.append((row[0], row[2], row[4], row[5]))

# Remove the first element from crime_data
crime_data.pop(0)

# Print the first 10 records
print(crime_data[:10])

# Find the Months with the Highest Number of Crimes
# Using the crime_data list from the prior exercise, you'll answer a common question that arises when dealing with crime data: How many crimes are committed each month?
#
# Feel free to use the IPython Shell to explore the crime_data list - it has been pre-loaded for you. For example, crime_data[0][0] will show you the first column of the first row which, in this case, is the date and time time that the crime occurred.

# Import Counter from collections and datetime from datetime.
# Create a Counter object called crimes_by_month.
# Loop over the crime_data list:
# Using the datetime.strptime() function, convert the first element of each item into a Python Datetime Object called date.
# Increment the counter for the month associated with this row by one. You can access the month of date using date.month.
# Print the 3 most common months for crime.

# Import necessary modules
from collections import Counter
from datetime import datetime

# Create a Counter Object: crimes_by_month
crimes_by_month = Counter()

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element of each item into a Python Datetime Object: date
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')

    # Increment the counter for the month of the row by one
    crimes_by_month[date.month] += 1

# Print the 3 most common months for crime
print(crimes_by_month.most_common(3))

# Transforming your Data Containers to Month and Location
# Now let's flip your crime_data list into a dictionary keyed by month with a list of location values for each month, and filter down to the records for the year 2016. Remember you can use the shell to look at the crime_data list, such as crime_data[1][4] to see the location of the crime in the second item of the list (since lists start at 0).

# Import defaultdict from collections and datetime from datetime.
# Create a dictionary that defaults to a list called locations_by_month.
# Loop over the crime_data list:
# Convert the first element to a date object exactly like you did in the previous exercise.
# If the year is 2016, set the key of locations_by_month to be the month of date and append the location (fifth element of row) to the values list.
# Print the dictionary. This has been done for you, so hit 'Submit Answer' to see the result!

# Import necessary modules
from collections import defaultdict
from datetime import datetime

# Create a dictionary that defaults to a list: locations_by_month
locations_by_month = defaultdict(list)

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element to a date object
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')

    # If the year is 2016
    if date.year == 2016:
        # Set the dictionary key to the month and add the location (fifth element) to the values list
        locations_by_month[date.month].append(row[4])

# Print the dictionary
print(locations_by_month)

# Find the Most Common Crimes by Location Type by Month in 2016
# Using the locations_by_month dictionary from the prior exercise, you'll now determine common crimes by month and location type. Because your dataset is so large, it's a good idea to use Counter to look at an aspect of it in an easier to manageable size and learn more about it.

# Import Counter from collections.
# Loop over the items from your dictionary, using tuple expansion to unpack locations_by_month.items() into month and locations.
# Make a Counter of the locations called location_count.
# Print the month.
# Print the five most common crime locations.

# Import Counter from collections
from collections import Counter

# Loop over the items from locations_by_month using tuple expansion of the month and locations
for month, locations in locations_by_month.items():
    # Make a Counter of the locations
    location_count = Counter(locations)
    # Print the month
    print(month)
    # Print the most common location
    print(location_count.most_common(5))

# Reading your Data with DictReader and Establishing your Data Containers
# Your data file, crime_sampler.csv contains in positional order: the date, block where it occurred, primary type of the crime, description of the crime, description of the location, if an arrest was made, was it a domestic case, and city district.
#
# You'll now use a DictReader to load up a dictionary to hold your data with the district as the key and the rest of the data in a list. The csv, defaultdict, and datetime modules have already been imported for you.

# Create a Python file object in read mode for crime_sampler.csv called csvfile.
# Create a dictionary that defaults to a list called crimes_by_district.
# Loop over a DictReader of the CSV file:
# Pop 'District' from each row and store it as district.
# Append the rest of the data (row) to the district key of crimes_by_district.

# Create the CSV file: csvfile
csvfile = open('crime_sampler.csv', 'r')

# Create a dictionary that defaults to a list: crimes_by_district
crimes_by_district = defaultdict(list)

# Loop over a DictReader of the CSV file
for row in csv.DictReader(csvfile):
    # Pop the district from each row: district
    district = row.pop('District')
    # Append the rest of the data to the list for proper district in crimes_by_district
    crimes_by_district[district].append(row)

# Determine the Arrests by District by Year
# Using your crimes_by_district dictionary from the previous exercise, you'll now determine the number arrests in each City district for each year. Counter is already imported for you. You'll want to use the IPython Shell to explore the crimes_by_district dictionary to determine how to check if an arrest was made.

# Loop over the crimes_by_district dictionary, unpacking it into the variables district and crimes.
# Create an empty Counter object called year_count.
# Loop over the crimes:
# If there was an arrest,
# Convert crime['Date'] to a datetime object called year.
# Add the crime to the Counter for the year, by using year as the key of year_count.
# Print the Counter. This has been done for you, so hit 'Submit Answer' to see the result!

# Loop over the crimes_by_district using expansion as district and crimes
for district, crimes in crimes_by_district.items():
    # Print the district
    print(district)

    # Create an empty Counter object: year_count
    year_count = Counter()

    # Loop over the crimes:
    for crime in crimes:
        # If there was an arrest
        if crime['Arrest'] == 'true':
            # Convert the Date to a datetime and get the year
            year = datetime.strptime(crime['Date'], '%m/%d/%Y %I:%M:%S %p').year
            # Increment the Counter for the year
            year_count[year] += 1

    # Print the counter
    print(year_count)

# Unique Crimes by City Block
# You're in the home stretch!
#
# Here, your data has been reshaped into a dictionary called crimes_by_block in which crimes are listed by city block. Your task in this exercise is to get a unique list of crimes that have occurred on a couple of the blocks that have been selected for you to learn more about. You might remember that you used set() to solve problems like this in Chapter 1.
#
# Go for it!

# Create a unique list of crimes for the '001XX N STATE ST' block called n_state_st_crimes and print it.
# Create a unique list of crimes for the '0000X W TERMINAL ST' block called w_terminal_st_crimes and print it.
# Find the crimes committed on 001XX N STATE ST but not 0000X W TERMINAL ST. Store the result as crime_differences and print it.

# Create a unique list of crimes for the first block: n_state_st_crimes
n_state_st_crimes = set(crimes_by_block['001XX N STATE ST'])

# Print the list
print(n_state_st_crimes)

# Create a unique list of crimes for the second block: w_terminal_st_crimes
w_terminal_st_crimes = set(crimes_by_block['0000X W TERMINAL ST'])

# Print the list
print(w_terminal_st_crimes)

# Find the differences between the two blocks: crime_differences
crime_differences = n_state_st_crimes.difference(w_terminal_st_crimes)

# Print the differences
print(crime_differences)

