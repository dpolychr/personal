# Creating a SparkSession
# We've already created a SparkSession for you called spark, but what if you're not sure there already is one? Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the SparkSession.builder.getOrCreate() method. This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary!

# Import SparkSession from pyspark.sql.
# Make a new SparkSession called my_spark using SparkSession.builder.getOrCreate().
# Print my_spark to the console to verify it's a SparkSession.

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)

# Viewing tables
# Once you've created a SparkSession, you can start poking around to see what data is in your cluster!
#
# Your SparkSession has an attribute called catalog which lists all the data inside the cluster. This attribute has a few methods for extracting different pieces of information.
#
# One of the most useful is the .listTables() method, which returns the names of all the tables in your cluster as a list.

# See what tables are in your cluster by calling spark.catalog.listTables() and printing the result!

# Print the tables in the catalog
print(spark.catalog.listTables())

# Are you query-ious?
# One of the advantages of the DataFrame interface is that you can run SQL queries on the tables in your Spark cluster. If you don't have any experience with SQL, don't worry, we'll provide you with queries! (To learn more SQL, start with our Introduction to SQL course.)
#
# As you saw in the last exercise, one of the tables in your cluster is the flights table. This table contains a row for every flight that left Portland International Airport (PDX) or Seattle-Tacoma International Airport (SEA) in 2014 and 2015.
#
# Running a query on this table is as easy as using the .sql() method on your SparkSession. This method takes a string containing the query and returns a DataFrame with the results!
#
# If you look closely, you'll notice that the table flights is only mentioned in the query, not as an argument to any of the methods. This is because there isn't a local object in your environment that holds that data, so it wouldn't make sense to pass the table as an argument.
#
# Remember, we've already created a SparkSession called spark in your workspace. (It's no longer called my_spark because we created it for you!)

# Use the .sql() method to get the first 10 rows of the flights table and save the result to flights10. The variable query contains the appropriate SQL query.
# Use the DataFrame method .show() to print flights10.

# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()

# Pandafy a Spark DataFrame
# Suppose you've run a query on your huge dataset and aggregated it down to something a little more manageable.
#
# Sometimes it makes sense to then take that table and work with it locally using a tool like pandas. Spark DataFrames make that easy with the .toPandas() method. Calling this method on a Spark DataFrame returns the corresponding pandas DataFrame. It's as simple as that!
#
# This time the query counts the number of flights to each airport from SEA and PDX.
#
# Remember, there's already a SparkSession called spark in your workspace!

# Run the query using the .sql() method. Save the result in flight_counts.
# Use the .toPandas() method on flight_counts to create a pandas DataFrame called pd_counts.
# Print the .head() of pd_counts to the console.

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())

# Put some Spark in your data
# In the last exercise, you saw how to move data from Spark to pandas. However, maybe you want to go the other direction, and put a pandas DataFrame into a Spark cluster! The SparkSession class has a method for this as well.
#
# The .createDataFrame() method takes a pandas DataFrame and returns a Spark DataFrame.
#
# The output of this method is stored locally, not in the SparkSession catalog. This means that you can use all the Spark DataFrame methods on it, but you can't access the data in other contexts.
#
# For example, a SQL query (using the .sql() method) that references your DataFrame will throw an error. To access the data in this way, you have to save it as a temporary table.
#
# You can do this using the .createTempView() Spark DataFrame method, which takes as its only argument the name of the temporary table you'd like to register. This method registers the DataFrame as a table in the catalog, but as this table is temporary, it can only be accessed from the specific SparkSession used to create the Spark DataFrame.
#
# There is also the method .createOrReplaceTempView(). This safely creates a new temporary table if nothing was there before, or updates an existing table if one was already defined. You'll use this method to avoid running into problems with duplicate tables.
#
# Check out the diagram to see all the different ways your Spark data structures interact with each other.
#
# There's already a SparkSession called spark in your workspace, numpy has been imported as np, and pandas as pd
#
# The code to create a pandas DataFrame of random numbers has already been provided and saved under pd_temp.
# Create a Spark DataFrame called spark_temp by calling the .createDataFrame() method with pd_temp as the argument.
# Examine the list of tables in your Spark cluster and verify that the new DataFrame is not present. Remember you can use spark.catalog.listTables() to do so.
# Register spark_temp as a temporary table named "temp" using the .createOrReplaceTempView() method. Rememeber that the table name is set including it as the only argument!
# Examine the list of tables again!

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())

# Dropping the middle man
# Now you know how to put data into Spark via pandas, but you're probably wondering why deal with pandas at all? Wouldn't it be easier to just read a text file straight into Spark? Of course it would!
#
# Luckily, your SparkSession has a .read attribute which has several methods for reading different data sources into Spark DataFrames. Using these you can create a DataFrame from a .csv file just like with regular pandas DataFrames!
#
# The variable file_path is a string with the path to the file airports.csv. This file contains information about different airports all over the world.
#
# A SparkSession named spark is available in your workspace.

# Use the .read.csv() method to create a Spark DataFrame called airports
# The first argument is file_path
# Pass the argument header=True so that Spark knows to take the column names from the first line of the file.
# Print out this DataFrame by calling .show().

# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()

# Creating columns
# In this chapter, you'll learn how to use the methods defined by Spark's DataFrame class to perform common data operations.
#
# Let's look at performing column-wise operations. In Spark you can do this using the .withColumn() method, which takes two arguments. First, a string with the name of your new column, and second the new column itself.
#
# The new column must be an object of class Column. Creating one of these is as easy as extracting a column from your DataFrame using df.colName.
#
# Updating a Spark DataFrame is somewhat different than working in pandas because the Spark DataFrame is immutable. This means that it can't be changed, and so columns can't be updated in place.
#
# Thus, all these methods return a new DataFrame. To overwrite the original DataFrame you must reassign the returned DataFrame using the method like so:
#
# df = df.withColumn("newCol", df.oldCol + 1)
# The above code creates a DataFrame with the same columns as df plus a new column, newCol, where every entry is equal to the corresponding entry from oldCol, plus one.
#
# To overwrite an existing column, just pass the name of the column as the first argument!
#
# Remember, a SparkSession called spark is already in your workspace.

# Use the spark.table() method with the argument "flights" to create a DataFrame containing the values of the flights table in the .catalog. Save it as flights.
# Print the output of flights.show(). The column air_time contains the duration of the flight in minutes.
# Update flights to include a new column called duration_hrs, that contains the duration of each flight in hours.

# Create the DataFrame flights
flights = spark.table("flights")

# Show the head
print(flights.show())

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)

# Filtering Data
# Now that you have a bit of SQL know-how under your belt, it's easier to talk about the analogous operations using Spark DataFrames.
#
# Let's take a look at the .filter() method. As you might suspect, this is the Spark counterpart of SQL's WHERE clause. The .filter() method takes either a Spark Column of boolean (True/False) values or the WHERE clause of a SQL expression as a string.
#
# For example, the following two expressions will produce the same output:
#
# flights.filter(flights.air_time > 120).show()
# flights.filter("air_time > 120").show()
# Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.

# Use the .filter() method to find all the flights that flew over 1000 miles two ways:
# First, pass a SQL string to .filter() that checks the distance is greater than 1000. Save this as long_flights1.
# Then pass a boolean column to .filter() that checks the same thing. Save this as long_flights2.
# Print the .show() of both DataFrames and make sure they're actually equal!

# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")

# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)

# Examine the data to check they're equal
print(long_flights1.show())
print(long_flights2.show())

# Selecting
# The Spark variant of SQL's SELECT is the .select() method. This method takes multiple arguments - one for each column you want to select. These arguments can either be the column name as a string (one for each column) or a column object (using the df.colName syntax). When you pass a column object, you can perform operations like addition or subtraction on the column to change the data contained in it, much like inside .withColumn().
#
# The difference between .select() and .withColumn() methods is that .select() returns only the columns you specify, while .withColumn() returns all the columns of the DataFrame in addition to the one you defined. It's often a good idea to drop columns you don't need at the beginning of an operation so that you're not dragging around extra data as you're wrangling. In this case, you would use .select() and not .withColumn().
#
# Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.

# Select the columns tailnum, origin, and dest from flights by passing the column names as strings. Save this as selected1.
# Select the columns origin, dest, and carrier using the df.colName syntax and then filter the result using both of the filters already defined for you (filterA and filterB) to only keep flights from SEA to PDX. Save this as selected2.

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size=size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

# Bootstrap replicates of the mean and the SEM
# In this exercise, you will compute a bootstrap estimate of the probability density function of the mean annual rainfall at the Sheffield Weather Station. Remember, we are estimating the mean annual rainfall we would get if the Sheffield Weather Station could repeat all of the measurements from 1883 to 2015 over and over again. This is a probabilistic estimate of the mean. You will plot the PDF as a histogram, and you will see that it is Normal.
#
# In fact, it can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be Normally distributed. (This does not hold in general, just for the mean and a few other statistics.) The standard deviation of this distribution, called the standard error of the mean, or SEM, is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set, sem = np.std(data) / np.sqrt(len(data)). Using hacker statistics, you get this same result without the need to derive it, but you will verify this result from your bootstrap replicates.
#
# The dataset has been pre-loaded for you into an array called rainfall

# Draw 10000 bootstrap replicates of the mean annual rainfall using your draw_bs_reps() function and the rainfall array. Hint: Pass in np.mean for func to compute the mean.
# As a reminder, draw_bs_reps() accepts 3 arguments: data, func, and size.
# Compute and print the standard error of the mean of rainfall.
# The formula to compute this is np.std(data) / np.sqrt(len(data)).
# Compute and print the standard deviation of your bootstrap replicates bs_replicates.
# Make a histogram of the replicates using the normed=True keyword argument and 50 bins.
# Hit 'Submit Answer' to see the plot!

