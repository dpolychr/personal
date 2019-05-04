import sys # to use argv
from urllib.request import urlopen

def fetch_words(url):
    with urlopen(url) as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)
    return story_words


def print_items(items):
    for item in items:
        print(item)

# Special attributes are delimited by double underscores

# __name__
# Evaluates to "__main__" or the actual module name depending on how the enclosing module is being used

# When importing we'd rather get the words as a list but when running directly we'd prefer the words to be printed


def main(url):
    # url = sys.argv[1] # get the second argument, with an index of 1, from the list
    words = fetch_words(url)
    print_items(words)

if __name__ == '__main__':
    main(sys.argv[1])

# The id keyword returns a unique identifier for an object
a = 496
id(a)
b = 1729
id(b)
b = a
id(b)
id(a) == id(b)
a is b
a is None

# Check using mutable objects like lists
r = [2, 4, 6]
r
s = r
s[1] = 17
s
s is r

m = [9, 15, 24]
def modify(k):
    k.append(39)
    print("k=", k)

def banner(message, border = '-'):
    line = border * len(message)
    print(line)
    print(message)
    print(line)

# Only use immuatable objects such as integers or strings for default values
def add_spam(menu=None):
    if menu is None:
        menu = []
    menu.append('spam')
    return menu

# Python is not going to coerce one type to another, the exception being the conversion to bool used for if statements and while-loop predicates

# Tuples
t = ("Norway", 4.953, 3)
t[0]
t[2]

# tuples can contain any type of object; nested tuples
a = ((220, 284), (1184, 1210), (2620, 2924), (5020, 5564))

# Single element tuple
a = (10,)

# In many cases eht parentheses of literal tuples may be omitted
p = 1, 1, 1, 4, 6, 19

# Tuples are useful for multiple return values
# Tuple unpacking allows us to destructure directly into named references

def minmax(items):
    return min(items), max(items)

minmax([83, 33, 84, 32, 85, 31, 86])

lower, upper = minmax([83, 33, 84, 32, 85, 31, 86])

# a, b = b, a is the idiomatic Python swap

a = 'jelly'
b = 'bean'
a, b = b, a

# use the tuple(iterable) constructor to create tuples from other iterable series of objects
tuple([561, 1105, 1729, 2465])
tuple("Carmichael")

5 in (3, 5, 17, 257, 65537)

# For joining large number of strings, the .join method should be preferred
s = "New"
s += "found"
s += "land"

# Call the join() method on the separator string
colors = ';'.join(['#45ff23', '#2321fa', '#1298a3', '#a32312'])

# We can then split them up again using the split method
colors.split(';')

# Without an argument, split() divides on whitespace
# join()-ing on an empty separator is an important and fast way of concatenating a collection of strings

''.join(['high', 'way', 'man'])

# The partition() method divided a string into three around a separator: prefix, separator, suffix. Returns a tuple so tuple unpucking is useful to destructure the result
"unforgetable".partition("forget")

departure, separator, arrival = "London:Edinburgh".partition(':')

# If we are not interested in capturing the separator value, you might see the "_" variable name used
origin, _, destination = "Boston-London".partition('-')

# Use format to insert values into strings
"The age of {0} is {1}".format('Jim', 32)

"Reticulating spline {} of {}. ".format(4, 23)

import math
"Math Constants: pi={m.pi}, e={m.e}".format(m=math)

# Optional third step value
list(range(0, 10, 2))

range(5) # If only one argument, it is the stop value

# If for some reason you need a counter, you should use the built-in enumerate function, which returns an iterable series of pairs, each pair being a tuple

t = [6, 372, 8862, 148800, 2096886]

for p in enumerate(t):
    print(p)

# Often combined with tuple unpacking
for i, v in enumerate(t):
    print("i = {}, v = {}".format(i, v))

s = "show how to index into sequences".split()

# To get the 1st and the last element
s[1:-1]

s[:3]

full_slice = s[:]

# Other more readable ways of copying a list, such as the .copy() method
u = s.copy()

v = list(s)

w = "the quick brown fox jumps over the lazy dog".split()

i = w.index("fox")

w[i]

# index(item) returns the integer index of the first equivalent element raises
# ValueError if not Found

# Count matching elements
w.count('the')

# The in and not in operators test for membership

# del seq[index] to remove by index
# seq.remove(item) to remove by value; raises ValueError if not present

a = "I accidentally the whole universe".split()

a.insert(2, "destroyed")

' '.join(a)

m = [1, 2, 3]

n = [4, 7, 11]

k = m + n

# In-place extenstion with += operator or extend() method
k += [18, 29, 47]

k.extend([76, 129, 199])

g = [1, 11, 21, 1211, 112111]

# k.sort() sorts in place
d = [5, 17, 41, 29, 71, 149, 3299, 7, 13, 67]

d.sort()

d.sort(reverse=True)
# key argument to sort() method accepts a function for producing a sort key from an item

h = 'not perplexing do handwriting family where I illegibly know doctors'. split()

h.sort(key=len)

' '.join(h)

x = [4, 9, 2, 1]

y = sorted(x)

# reversed() built-in function reverses any iterable series

p = [9, 3, 1, 0]

q = reversed(p)

# reversed() returns an iterator so we'd need to use list() to get the result

list(q)

# dict: lies in the heart of many python programs including the Python interpreter itself

urls = {'Google': 'http://google.com',
        'Pluralsight': 'https://pluralsight.con',
        'Microsoft': 'http://microsoft.com'}

for key in urls.keys():
    print(key)

# Use items() for an iterable view onto the series of key-value tuples
for key, value in urls.items():
    print(key, value)

for key, value in urls.items():
    print("{key} => {value}".format(key = key, value = value))

p = {6, 28, 496, 8128, 33550336}

# set (removing elements)
# remove(item) requires that item is present, otherwise raises KeyError
# discard(item) always succeeds

