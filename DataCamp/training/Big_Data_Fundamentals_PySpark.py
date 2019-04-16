# Apache Spark is written in Scala

# Pyspark APIs are similar to Pandas and scikit-learn

# Understanding SparkContext
# A SparkContext represents the entry point to Spark functionality. It's like a key to your car. PySpark automatically creates a SparkContext for you in the PySpark shell (so you don't have to create it by yourself) and is exposed via a variable sc.
#
# In this simple exercise, you'll find out the attributes of the SparkContext in your PySpark shell which you'll be using for the rest of the course.

# Print the version of SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)

# Print the Python version of SparkContext
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)

# Print the master of SparkContext
print("The master of Spark Context in the PySpark shell is", sc.master)

# Interactive Use of PySpark
# PySpark shell is an interactive shell for basic testing and debugging but it is quite powerful. The easiest way to demonstrate the power of PySpark’s shell is to start using it. In this example, you'll load a simple list containing numbers ranging from 1 to 100 in the PySpark shell.
#
# The most important thing to understand here is that we are not creating any SparkContext object because PySpark automatically creates the SparkContext object named sc, by default in the PySpark shell.

# Create a python list named numb containing the numbers 1 to 100.
# Load the list into Spark using Spark Context's parallelize method and assign it to a variable spark_data.

# Create a python list of numbers from 1 to 100
numb = range(1, 100)

# Load the list into PySpark
spark_data = sc.parallelize(numb)

# Loading data in PySpark shell
# In PySpark, we express our computation through operations on distributed collections that are automatically parallelized across the cluster. In the previous exercise, you have seen an example of loading a list as parallelized collections and in this exercise, you'll load the data from a local file in PySpark shell.
#
# Remember you already have a SparkContext sc and file_path variable (which is the path to the README.md file) already available in your workspace.

# Load a local file into PySpark shell
lines = sc.textFile(file_path)

# Add the number 2 to all the items in a list
items = [1, 2, 3, 4]
list(map(lambda x: x + 2, items))

# filter() function takes a fun and a list and returns a new list for which the fun evaluates as true

# Example of filter to filter out only odd numbers from a list
items = [1, 2, 3, 4]
list(filter(lambda x: (x%2 != 0), items))

# Use of lambda() with map()
# The map() function in Python returns a list of the results after applying the given function to each item of a given iterable (list, tuple etc.). The general syntax of map() function is map(fun, iter). We can also use lambda functions with map(). The general syntax of map() function with lambda() is map(lambda <agument>:<expression>, iter). Refer to slide 5 of video 1.7 for general help of map() function with lambda().
#
# In this exercise, you'll be using lambda function inside the map() built-in function to square all numbers in the list.

# Print my_list which is available in your environment.
# Square each item in my_list using map() and lambda().
# Print the result of map function.

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list
squared_list_lambda = list(map(lambda x: x ** 2, my_list))

# Print the result of the map function
print("The squared numbers are", squared_list_lambda)

# Use of lambda() with filter()
# Another function that is used extensively in Python is the filter() function. The filter() function in Python takes in a function and a list as arguments. The general syntax of filter() function is filter(function, list_of_input). Similar to map(), filter() can be used with lambda() function. The general syntax of filter() function with lambda() is filter(lambda <argument>:<expression>, list). Refer to slide 6 of video 1.7 for general help of filter() function with lambda().
#
# In this exercise, you'll be using lambda() function inside the filter() built-in function to find all the numbers divisible by 10 in the list.

# Print my_list2 which is available in your environment.
# Filter the numbers divisible by 10 from my_list2 using filter() and lambda().
# Print the numbers divisible by 10 from my_list2.

my_list2 = [10, 21, 31, 40, 51, 60, 72, 80, 93, 101]

# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)

# RDD: Resilient Distributed Datasets. Collection of data distributed across the cluster

# Creating RDDs:

# a. Parallelizing an existing collection of objects

# b. External datasets: i) Files in HDFS, ii) Objects in amazon S3 bucket, iii) lines in a text file

# RDDs from Parallelized collections
# Resilient Distributed Dataset (RDD) is the basic abstraction in Spark. It is an immutable distributed collection of objects. Since RDD is a fundamental and backbone data type in Spark, it is important that you understand how to create it. In this exercise, you'll create your first RDD in PySpark from a collection of words.
#
# Remember you already have a SparkContext sc available in your workspace.

# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))

# RDDs from External Datasets
# PySpark can easily create RDDs from files that are stored in external storage devices such as HDFS (Hadoop Distributed File System), Amazon S3 buckets, etc. However, the most common method of creating RDD's is from files stored in your local file system. This method takes a file path and reads it as a collection of lines. In this exercise, you'll create an RDD from the file path (file_path) with the file name README.md which is already available in your workspace.
#
# Remember you already have a SparkContext sc available in your workspace.

# Print the file_path in the PySpark shell.
# Create an RDD named fileRDD from a file_path with the file name README.md.
# Print the type of the fileRDD created.

# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

# Partitions in your data
# SparkContext's textFile() method takes an optional second argument called minPartitions for specifying the minimum number of partitions. In this exercise, you'll create an RDD named fileRDD_part with 5 partitions and then compare that with fileRDD that you created in the previous exercise. Refer to the "Understanding Partition" slide in video 2.1 to know the methods for creating and getting the number of partitions in an RDD.
#
# Remember, you already have a SparkContext sc, file_path and fileRDD available in your workspace.

# Find the number of partitions that support fileRDD RDD.
# Create an RDD named fileRDD_part from the file path but create 5 partitions.
# Confirm the number of partitions in the new fileRDD_part RDD.

# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

