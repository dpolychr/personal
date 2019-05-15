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


