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

