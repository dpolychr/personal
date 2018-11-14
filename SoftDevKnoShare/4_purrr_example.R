# This material have been taken from the Datacamp course: Foundations of Functional Programming with purrr 
# https://www.datacamp.com/courses/foundations-of-functional-programming-with-purrr

# Set working directory
setwd("~/git/personal/SoftDevKnoShare/Data/")

# Read csvs
toycsv1 <- read_csv("swimming_pools.csv") 
toycsv2 <- read_csv("mtcars.csv") 
toycsv3 <- read_csv("iris.csv")

# List files
toycsvs <- list.files(pattern=".csv") 

list_of_toycsvs <- list()
for(i in toycsvs){ list_of_toycsvs[[i]] <- read.csv(i) }

# Iteration without purrr
files <- list.files()

d <- list()
# Loop through the values 1 through 10, to add them to d
for(i in 1:3){
  d[[i]] <- read_csv(files[i]) }

# Iteration with purrr
# map(object, function)
# object - can be a vector or a list
# function - any function in R that takes the input offered by the object
d <- map(files, read_csv)

# Demonstrate type consistency
# sapply() is a common offender returning unstable types. The type of output returned from sapply() depends on the type of input

df <- data.frame(
  a = 1L,
  b = 1.5,
  y = Sys.time(),
  z = ordered(1)
)

A <- sapply(df[1:4], class) 
B <- sapply(df[3:4], class)

# A will be a list, B will be a character matrix. This unpredictable behaviour is a sign that you shouldn't rely on sapply() inside your own functions.

# So, what do you do? Use alternate functions that are type consistent! And you already know a whole set: the map() functions in purrr.

# In this example, when we call class() on the columns of the data frame we are expecting character output, so our function of choice should be: map_chr():

df <- data.frame(
  a = 1L,
  b = 1.5,
  y = Sys.time(),
  z = ordered(1)
)

A <- map_chr(df[1:4], class) 
B <- map_chr(df[3:4], class)

# Except that gives us errors. This is a good thing! It alerts us that our assumption (that class() would return purely character output) is wrong.

# Let's look at a couple of solutions. First, we could use map() instead of map_chr(). Our result will always be a list, no matter the input.

# sapply calls
A <- sapply(df[1:4], class) 
B <- sapply(df[3:4], class)
C <- sapply(df[1:2], class) 

# Demonstrate type inconsistency
str(A)
str(B)
str(C)

# Use map() to define X, Y and Z
X <- map(df[1:4], class) 
Y <- map(df[3:4], class)
Z <- map(df[1:2], class) 

# Use str() to check type consistency
str(X)
str(Y)
str(Z)

# Demonstration of pluck
dfs <- list(iris, mtcars)
dfs %>% pluck(2, attr_getter("row.names"))

# Summarize the columns of a row using purrrlyr: Tools at the Intersection of 'purrr' and 'dplyr'
mtcars %>%
  select(am, gear, carb) %>%
  purrrlyr::by_row(sum, .collate = "cols", .to = "sum_am_gear_carb") -> mtcars2
head(mtcars2)

# purrr's Lift function
x <- list(x = c(1:100, NA, 1000), na.rm = TRUE, trim = 0.9)
lift_dl(mean)(x)

# Or in a pipe:
mean %>% lift_dl() %>% invoke(x)

# Load the data
data(gh_users)

# Check if data has names
names(gh_users)

# Map over name element of list
map(gh_users, ~.x[["name"]])

# Setting names
# Setting list names makes working with lists much easier in many scenarios; it makes the code easier to read, which is especially important when reviewing code weeks or months later.

# Here you are going to work with the gh_repos and gh_users datasets and set their names in two different ways. The two methods will give the same result: a list with named elements.

# Set the names on gh_users using the "name" element and use the map_*() function that outputs a character vector.
# Explore the structure of gh_repos to see where the owner info is stored.
# Set the names on gh_repos based on the owner of the repo, using the set_names() and map_*() functions.

# Name gh_users with the names of the users
gh_users <- gh_users %>% 
  set_names(map_chr(gh_users, "name"))

# Check gh_repos structure
str(gh_repos)

# Name gh_repos with the names of the repo owner 
gh_repos_named <- gh_repos %>% 
  map_chr(~map_chr(.x, ~.x$owner$login)[1]) %>% 
  set_names(gh_repos, .)

# Determine who joined github first
map_chr(gh_users, ~.x[["created_at"]]) %>%
  set_names(map_chr(gh_users, "name")) %>%
  sort()

# Determine who has the most public repositories
map_dbl(gh_users, ~.x[["public_repos"]]) %>%
  set_names(map_chr(gh_users, "name")) %>%
  sort()

# purrr and scatterplots
# Since ggplot() does not accept lists as an input, it can be paired up with purrr to go from a list to a dataframe to a ggplot() graph in just a few lines of code.
# 
# You will continue to work with the gh_users data for this exercise. You will use a map_*() function to pull out a few of the named elements and transform them into the correct datatype. Then create a scatterplot that compares the user's number of followers to the user's number of public repositories.
# 
# map() over gh_users, use the map_*() function that creates a dataframe, with four columns, named "login", "name", "followers" and "public_repos".
# Pipe that dataframe into a scatterplot, where the x axis is followers and y is public_repos.

# Create a dataframe with four columns
map_df(gh_users, `[`, 
       c("login", "name", "followers", "public_repos")) %>%
  # Plot followers by public_repos
  ggplot(., 
         aes(x = followers, y = public_repos)) + 
  # Create scatter plots
  geom_point()

# What is the distribution of heights of characters in each of the Star Wars films?
#   Different movies take place on different sets of planets, so you might expect to see different distributions of heights from the characters. Your first task is to transform the two datasets into dataframes since ggplot() requires a dataframe input. Then you will join them together, and plot the result, a histogram with a different facet, or subplot, for each film.

# Create a dataframe with the "title" of each film, and the "characters" from each film in the sw_films dataset.
# Create a dataframe with the "height", "mass", "name", and "url" elements from sw_people.
# Join the two dataframes together using the "characters" and "urls" keys.
# Create a ggplot() histogram with x = height, faceted by filmtitle.

# Turn data into correct dataframe format
film_by_character <- tibble(filmtitle = map_chr(sw_films, "title")) %>%
  transmute(filmtitle, characters = map(sw_films, "characters")) %>%
  unnest()

# Try 
tibble(filmtitle = map_chr(sw_films, "title")) %>% transmute(filmtitle, characters = map_chr(sw_films, "characters"))

# Why it does not work

# Pull out elements from sw_people
sw_characters <- map_df(sw_people, `[`, c("height", "mass", "name", "url"))

# Join the two new objects
inner_join(film_by_character, sw_characters, by = c("characters"= "url")) %>%
  # Make sure the columns are numbers
  mutate(height = as.numeric(height), mass = as.numeric(mass)) %>%
  ggplot(aes(x = height))+
  geom_histogram(stat = "count")+
  facet_wrap(~filmtitle)

# Simulating data with multiple inputs using map2()
# The map() function is great if you need to iterate over one list, however, you will often need to iterate over two lists at the same time. This is where map2() comes in. While map() takes the list as the .x argument; map2() takes two lists as two arguments: .x and .y.
# 
# To test out map2(), you are going to create a simple dataset, with one list of numbers and one list of strings. You will put these two lists together and create some simulated data.
# 
# Create a list, means, of the values 1 through 3.
# Create a sites list with "north", "west", and "east".
# map2() over the sites and means lists to create a dataframe with two columns.
# First column is sites; second column is generated by rnorm() with mean from the means list.

# List of 1 through 3
means <- list(1,2,3)

# Create sites list
sites <- list("north", "west", "east")

# Map over two arguments: years and mu
list_of_files_map2 <- map2(sites, means, ~data.frame(sites = .x,
                                                     a = rnorm(mean = .y, n = 200, sd = (5/2))))

list_of_files_map2

# What to do if things go wrong with lists?
  
# safely() result element and an error element

# safely() replace with NA
# If you map() over a list, and one of the elements does not have the right data type, you will not get the output you expect. Perhaps you are trying to do a mathematical operation on each element, and it turns out one of the elements is a character - it simply won't work.

# If you have a very large list, figuring out where things went wrong, and what exactly went wrong can be hard. That is where safely() comes in; it shows you both your results and where the errors occurred in your map() call.
# 
# Use safely() with log(). This will fail to work on -10, so we'll pipe it into transpose() to put the results first.
# 
# # Map safely over log
a <- list(-10, 1, 10, 0) %>% 
  map(safely(log, otherwise = NA_real_)) %>%
  # Transpose the result
  transpose()

# Print the list
a

# Print the result element in the list
a[["result"]]

# Print the error element in the list
a[["error"]]

# Convert data to numeric with purrr
# In the sw_people dataset, some of the Star Wars characters have unknown heights. If youe want to do some data exploration and determine how character height differs depending on their home planet, you need to write your code so that R understands the difference between heights and missing values. Currently, the missing values are entered as "unknown", but you would like them as NA. In this exercise, you will combine map() and ifelse() to fix this issue.

# Load the sw_people dataset.
# Map over sw_people and pull out "height".
# Then map over the output and if an element is labeled as "unknown" change it to NA, otherwise, convert the value into a number with as.numeric().

# Load sw_people data
data(sw_people)

# Map over sw_people and pull out the height element
height_cm <- map(sw_people, ~.x[["height"]]) %>%
  map(function(x){
    ifelse(x == "unknown",NA,
           as.numeric(x))
  })

# Finding the problem areas
# When you are working with a small list, it might not seem like a lot of work to go through things manually and figure out what element has an issue. But if you have a list with hundreds or thousands of elements, you want to automate that process.
# 
# Now you'll look at a situation with a larger list, where you can see how the error message can be useful to check through the entire list for issues.
# 
# map() over sw_people and pull out the "height" element.
# map() over safely() to convert the heights from centimeters into feet.
# Set quiet = FALSE so that errors are printed.
# Pipe into transpose(), to print the results first.

# Map over sw_people and pull out the height element
height_ft <- map(sw_people, ~.x[["height"]]) %>% 
  map(safely(function(x){
    x * 0.0328084
  }, quiet = FALSE)) %>% 
transpose

# Print your list, the result element, and the error element
height_ft
height_ft[["result"]]
height_ft[["error"]]

# possibly is going to do the same things safely did; it just gives a simpler output, with just the results and none of the error messages
# 
# Replace safely() with possibly()
# Once you have figured out how to solve an issue with safely(), (e.g., output an NA in place of an error), swap out safely() with possibly(). possibly() will run through your code and implement your desired changes without printing out the error messages.

# You'll now map() over log() again, but you will use possibly() instead of safely() since you already know how to resolve your errors.

# Create a list with the values -10, 1, 10, and 0.
# map() over this list to take the log() of each element, using possibly().
# Use NA_real_ to fix any elements that are not the right data type.


# Take the log of each element in the list
a <- list(-10, 1, 10, 0) %>% 
  map(possibly(function(x){
    log(x)
  },otherwise=NA_real_))

# Convert values with possibly()
# Let's say you need to convert the Star Wars character heights in sw_people from centimeters to feet. You already know that some of the heights have missing data, so you will use possibly() to convert missing values into NA. Then you will multiply each of the existing values by 0.0328084 to convert them from centimeters into feet.

# To get a feel for your data, print out height_cm in the console to check out the heights in centimeters.

# Pipe the height_cm object into a map_*() function that returns double vectors.
# Convert each element in height_cm into feet (multiply it by 0.0328084).
# Since not all elements are numeric, use possibly() to replace instances that do not work with NA_real_.

# Create a piped workflow that returns double vectors
height_cm %>%  
  map_dbl(possibly(function(x){
  # Convert centimeters to feet
    x * 0.0328084
}, otherwise=NA_real_)) 

