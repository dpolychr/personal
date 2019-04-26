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
