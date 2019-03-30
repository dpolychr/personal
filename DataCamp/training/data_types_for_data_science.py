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

