# SQLAlchemy

# Engines and Connection Strings
# Alright, it's time to create your first engine! An engine is just a common interface to a database, and the information it requires to connect to one is contained in a connection string, such as sqlite:///census_nyc.sqlite. Here, sqlite is the database driver, while census_nyc.sqlite is a SQLite file contained in the local directory.

# You can learn a lot more about connection strings in the SQLAlchemy documentation.

# Your job in this exercise is to create an engine that connects to a local SQLite file named census.sqlite. Then, print the names of the tables it contains using the .table_names() method. Note that when you just want to print the table names, you do not need to use engine.connect() after creating the engine.

# Import create_engine from the sqlalchemy module.
# Using the create_engine() function, create an engine for a local file named census.sqlite with sqlite as the driver. Be sure to enclose the connection string within quotation marks.
# Print the output from the .table_names() method on the engine.

# Import create_engine
from sqlalchemy import create_engine

# Create an engine that connects to the census.sqlite file: engine
engine = create_engine('sqlite:///census.sqlite')

# Print table names
print(engine.table_names())

# Autoloading Tables from a Database
# SQLAlchemy can be used to automatically load tables from a database using something called reflection. Reflection is the process of reading the database and building the metadata based on that information. It's the opposite of creating a Table by hand and is very useful for working with existing databases. To perform reflection, you need to import the Table object from the SQLAlchemy package. Then, you use this Table object to read your table from the engine and autoload the columns. Using the Table object in this manner is a lot like passing arguments to a function. For example, to autoload the columns with the engine, you have to specify the keyword arguments autoload=True and autoload_with=engine to Table().

# In this exercise, your job is to reflect the census table available on your engine into a variable called census. The metadata has already been loaded for you using MetaData() and is available in the variable metadata

# Import the Table object from sqlalchemy.
# Reflect the census table by using the Table object with the arguments:
# The name of the table as a string ('census').
# The metadata, contained in the variable metadata.
# autoload=True
# The engine to autoload with - in this case, engine.
# Print the details of census using the repr() function.

# Import Table
from sqlalchemy import Table

# Reflect census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print census table metadata
print(repr(census))

# Viewing Table Details
# Great job reflecting the census table! Now you can begin to learn more about the columns and structure of your table. It is important to get an understanding of your database by examining the column names. This can be done by using the .columns attribute and accessing the .keys() method. For example, census.columns.keys() would return a list of column names of the census table.

# Following this, we can use the metadata container to find out more details about the reflected table such as the columns and their types. For example, table objects are stored in the metadata.tables dictionary, so you can get the metadata of your census table with metadata.tables['census']. This is similar to your use of the repr() function on the census table from the previous exercise.

# Reflect the census table as you did in the previous exercise using the Table() function.
# Print a list of column names of the census table by applying the .keys() method to census.columns.
# Print the details of the census table using the metadata.tables dictionary along with the repr() function. To do this, first access the 'census' key of the metadata.tables dictionary, and place this inside the provided repr() function.

# Reflect the census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print the column names
print(census.columns.keys())

# Print full table metadata
print(repr(metadata.tables['census']))

# SQL to select data, insert new data, update existing data and delete it
# Create & Alter data

# Basic SQL querying
SELECT column_name FROM table_name

stmt = 'SELECT * FROM people'

# SQLAlchemy to Build Queries
# Pythinic way to build SQL statements
# Hides differences between backend database types

# Selecting data from a Table: raw SQL
# Using what we just learned about SQL and applying the .execute() method on our connection, we can leverage a raw SQL query to query all the records in our census table. The object returned by the .execute() method is a ResultProxy. On this ResultProxy, we can then use the .fetchall() method to get our results - that is, the ResultSet.

# In this exercise, you'll use a traditional SQL query. In the next exercise, you'll move to SQLAlchemy and begin to understand its advantages. Go for it!

# Build a SQL statement to query all the columns from census and store it in stmt. Note that your SQL statement must be a string.
# Use the .execute() and .fetchall() methods on connection and store the result in results. Remember that .execute() comes before .fetchall() and that stmt needs to be passed to .execute().
# Print results.

# Build select statement for census table: stmt
stmt = 'SELECT * FROM census'

# Execute the statement and fetch the results: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Selecting data from a Table with SQLAlchemy
# Excellent work so far! It's now time to build your first select statement using SQLAlchemy. SQLAlchemy provides a nice "Pythonic" way of interacting with databases. So rather than dealing with the differences between specific dialects of traditional SQL such as MySQL or PostgreSQL, you can leverage the Pythonic framework of SQLAlchemy to streamline your workflow and more efficiently query your data. For this reason, it is worth learning even if you may already be familiar with traditional SQL.

# In this exercise, you'll once again build a statement to query all records from the census table. This time, however, you'll make use of the select() function of the sqlalchemy module. This function requires a list of tables or columns as the only required argument.

# Table and MetaData have already been imported. The metadata is available as metadata and the connection to the database as connection.

# Import select from the sqlalchemy module.
# Reflect the census table. This code is already written for you.
# Create a query using the select() function to retrieve the census table. To do so, pass a list to select() containing a single element: census.
# Print stmt to see the actual SQL query being created. This code has been written for you.
# Using the provided print() function, print all the records from the census table. To do this:
# Use the .execute() method on connection with stmt as the argument to retrieve the ResultProxy.
# Use .fetchall() on connection.execute(stmt) to retrieve the ResultSet.

# Import select
from sqlalchemy import select

# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Build select statement for census table: stmt
stmt = select([census])

# Print the emitted statement to see the SQL emitted
print(stmt)

# Execute the statement and print the results
print(connection.execute(stmt).fetchall())

# Handling a ResultSet
# Recall the differences between a ResultProxy and a ResultSet:

# ResultProxy: The object returned by the .execute() method. It can be used in a variety of ways to get the data returned by the query.
# ResultSet: The actual data asked for in the query when using a fetch method such as .fetchall() on a ResultProxy.
# This separation between the ResultSet and ResultProxy allows us to fetch as much or as little data as we desire.

# Once we have a ResultSet, we can use Python to access all the data within it by column name and by list style indexes. For example, you can get the first row of the results by using results[0]. With that first row then assigned to a variable first_row, you can get data from the first column by either using first_row[0] or by column name such as first_row['column_name']. You'll now practice exactly this using the ResultSet you obtained from the census table in the previous exercise. It is stored in the variable results. Enjoy!

# Extract the first row of results and assign it to the variable first_row.
# Print the value of the first column in first_row.
# Print the value of the 'state' column in first_row.

# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)

# Print the first column of the first row by using an index
print(first_row[0])

# Print the 'state' column of the first row by using its name
print(first_row['state'])

# Select all the records for the state of California
stmt = select([census])

stmt = stmt.where(census.columns.state == 'California')

results = connection.execute(stmt).fetchall()

for result in results:
    print(result.state, result.age)

# More complex conditions than simple operators
# in_(), like(), between()

# Example
stmt = select([census])

stmt = stmt.where(
    census.columns.state.startswith('New'))

for result in connection.execute(stmt):
    print(result.state, result.pop2000)

# Conjunctions allow us to have multiple criteria in a where clause
# e.g. and_(), not_(), or_()

from sqlalchemy import or_

stmt = select([census])

stmt=stmt.where(
    or_(census.columns.state == 'California',
        census.columns.state == 'New York'
        )
)

# Connecting to a PostgreSQL Database
# In these exercises, you will be working with real databases hosted on the cloud via Amazon Web Services (AWS)!

# Let's begin by connecting to a PostgreSQL database. When connecting to a PostgreSQL database, many prefer to use the psycopg2 database driver as it supports practically all of PostgreSQL's features efficiently and is the standard dialect for PostgreSQL in SQLAlchemy.

# You might recall from Chapter 1 that we use the create_engine() function and a connection string to connect to a database.

# There are three components to the connection string in this exercise: the dialect and driver ('postgresql+psycopg2://'), followed by the username and password ('student:datacamp'), followed by the host and port ('@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/'), and finally, the database name ('census'). You will have to pass this string as an argument to create_engine() in order to connect to the database.

# Import create_engine from sqlalchemy.
# Create an engine to the census database by concatenating the following strings:
# 'postgresql+psycopg2://'
# 'student:datacamp'
# '@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com'
# ':5432/census'
# Use the .table_names() method on engine to print the table names.
# from sqlalchemy import create_engine
# engine = create_engine('postgresql+psycopg2://student:datacamp@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/census')

# Import create_engine function
from sqlalchemy import create_engine

# Create an engine to the census database
engine = create_engine('postgresql+psycopg2://student:datacamp@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/census')

# Use the .table_names() method on the engine to print the table names
print(engine.table_names())

# Filter data selected from a Table - Simple
# Having connected to the database, it's now time to practice filtering your queries!

# As mentioned in the video, a where() clause is used to filter the data that a statement returns. For example, to select all the records from the census table where the sex is Female (or 'F') we would do the following:

# select([census]).where(census.columns.sex == 'F')

# In addition to == we can use basically any python comparison operator (such as <=, !=, etc) in the where() clause.

# Select all records from the census table by passing in census as a list to select().
# Append a where clause to stmt to return only the records with a state of 'New York'.
# Execute the statement stmt using .execute() and retrieve the results using .fetchall().
# Iterate over results and print the age, sex and pop2008 columns from each record. For example, you can print out the age of result with result.age.

# Create a select query: stmt
stmt = select([census])

# Add a where clause to filter the results to only those for New York
stmt = stmt.where(census.columns.state == 'New York')

# Execute the query to retrieve all the data returned: results
results = connection.execute(stmt).fetchall()

# Loop over the results and print the age, sex, and pop2008
for result in results:
    print(result.age, result.sex, result.pop2008)

# Filter data selected from a Table - Expressions
# In addition to standard Python comparators, we can also use methods such as in_() to create more powerful where() clauses. You can see a full list of expressions in the SQLAlchemy Documentation.

# We've already created a list of some of the most densely populated states.

# Select all records from the census table by passing it in as a list to select().
# Append a where clause to return all the records with a state in the states list. Use in_(states) on census.columns.state to do this.
# Loop over the ResultProxy connection.execute(stmt) and print the state and pop2000 columns from each record.

# Create a query for the census table: stmt
stmt = select([census])

# Append a where clause to match all the states in_ the list states
stmt = stmt.where(census.columns.state.in_(states))

# Loop over the ResultProxy and print the state and its population in 2000
for result in connection.execute(stmt):
    print(result.state, result.pop2000)

# Filter data selected from a Table - Advanced
# You're really getting the hang of this! SQLAlchemy also allows users to use conjunctions such as and_(), or_(), and not_() to build more complex filtering. For example, we can get a set of records for people in New York who are 21 or 37 years old with the following code:

select([census]).where(
  and_(census.columns.state == 'New York',
       or_(census.columns.age == 21,
          census.columns.age == 37
         )
      )
  )

# Import and_ from the sqlalchemy module.
# Select all records from the census table.
# Append a where clause to filter all the records whose state is 'California', and whose sex is not 'M'.
# Iterate over the ResultProxy and print the age and sex columns from each record.

# Import and_
from sqlalchemy import and_

# Build a query for the census table: stmt
stmt = select([census])

# Append a where clause to select only non-male records from California using and_
stmt = stmt.where(
    # The state of California with a non-male sex
    and_(census.columns.state == 'California',
         census.columns.sex != 'M'
         )
)

# Loop over the ResultProxy printing the age and sex
for result in connection.execute(stmt):
    print(result.age, result.sex)

# Ordering by a Single Column
# To sort the result output by a field, we use the .order_by() method. By default, the .order_by() method sorts from lowest to highest on the supplied column. You just have to pass in the name of the column you want sorted to .order_by().

# In the video, for example, Jason used stmt.order_by(census.columns.state) to sort the result output by the state column.

# Select all records of the state column from the census table. To do this, pass census.columns.state as a list to select().
# Append an .order_by() to sort the result output by the state column.
# Execute stmt using the .execute() method on connection and retrieve all the results using .fetchall().
# Print the first 10 rows of results.

# Build a query to select the state column: stmt
stmt = select([census.columns.state])

# Order stmt by the state column
stmt = stmt.order_by('state')

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the first 10 results
print(results[:10])

# Ordering in Descending Order by a Single Column
# You can also use .order_by() to sort from highest to lowest by wrapping a column in the desc() function. Although you haven't seen this function in action, it generalizes what you have already learned.

# Pass desc() (for "descending") inside an .order_by() with the name of the column you want to sort by. For instance, stmt.order_by(desc(table.columns.column_name)) sorts column_name in descending order.

# Import desc from the sqlalchemy module.
# Select all records of the state column from the census table.
# Append an .order_by() to sort the result output by the state column in descending order. Save the result as rev_stmt.
# Execute rev_stmt using connection.execute() and fetch all the results with .fetchall(). Save them as rev_results.
# Print the first 10 rows of rev_results

# Import desc
from sqlalchemy import desc

# Build a query to select the state column: stmt
stmt = select([census.columns.state])

# Order stmt by state in descending order: rev_stmt
rev_stmt = stmt.order_by(desc('state'))

# Execute the query and store the results: rev_results
rev_results = connection.execute(rev_stmt).fetchall()

# Print the first 10 rev_results
print(rev_results[:10])

# Ordering by Multiple Columns
# We can pass multiple arguments to the .order_by() method to order by multiple columns. In fact, we can also sort in ascending or descending order for each individual column. Each column in the .order_by() method is fully sorted from left to right. This means that the first column is completely sorted, and then within each matching group of values in the first column, it's sorted by the next column in the .order_by() method. This process is repeated until all the columns in the .order_by() are sorted.

# Select all records of the state and age columns from the census table.
# Use .order_by() to sort the output of the state column in ascending order and age in descending order. (NOTE: desc is already imported).
# Execute stmt using the .execute() method on connection and retrieve all the results using .fetchall().
# Print the first 20 results.

# Build a query to select state and age: stmt
stmt = select([census.columns.state, census.columns.age])

# Append order by to ascend by state and descend by age
stmt = stmt.order_by(census.columns.state, desc(census.columns.age))

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print the first 20 results
print(results[:20])

from sqlalchemy import func

stmt = select([func.sum(census.columns.pop2008)])

results = connection.execute(stmt).scalar() # scalar fetch method to get back just a value and print it

# Important not to import the sum function directly because it will conflict with Python's builtin sum function

# Counting Distinct Data
# As mentioned in the video, SQLAlchemy's func module provides access to built-in SQL functions that can make operations like counting and summing faster and more efficient.

# In the video, Jason used func.sum() to get a sum of the pop2008 column of census as shown below:

select([func.sum(census.columns.pop2008)])
# If instead you want to count the number of values in pop2008, you could use func.count() like this:

select([func.count(census.columns.pop2008)])
# Furthermore, if you only want to count the distinct values of pop2008, you can use the .distinct() method:

select([func.count(census.columns.pop2008.distinct())])
# In this exercise, you will practice using func.count() and .distinct() to get a count of the distinct number of states in census.

# So far, you've seen .fetchall() and .first() used on a ResultProxy to get the results. The ResultProxy also has a method called .scalar() for getting just the value of a query that returns only one row and column.

# This can be very useful when you are querying for just a count or sum.

# Build a select statement to count the distinct values in the state field of census.
# Execute stmt to get the count and store the results as distinct_state_count.
# Print the value of distinct_state_count.

# Build a query to count the distinct states values: stmt
stmt = select([func.count(census.columns.state.distinct())])

# Execute the query and store the scalar result: distinct_state_count
distinct_state_count = connection.execute(stmt).scalar()

# Print the distinct_state_count
print(distinct_state_count)

# Count of Records by State
# Often, we want to get a count for each record with a particular value in another column. The .group_by() method helps answer this type of query. You can pass a column to the .group_by() method and use in an aggregate function like sum() or count(). Much like the .order_by() method, .group_by() can take multiple columns as arguments.

# Import func from sqlalchemy.
# Build a select statement to get the value of the state field and a count of the values in the age field, and store it as stmt.
# Use the .group_by() method to group the statement by the state column.
# Execute stmt using the connection to get the count and store the results as results.
# Print the keys/column names of the results returned using results[0].keys()

# Import func
from sqlalchemy import func

# Build a query to select the state and count of ages by state: stmt
stmt = select([census.columns.state, func.count(census.columns.age)])

# Group stmt by state
stmt = stmt.group_by('state')

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())

# Determining the Population Sum by State
# To avoid confusion with query result column names like count_1, we can use the .label() method to provide a name for the resulting column. This gets appendedto the function method we are using, and its argument is the name we want to use.

# We can pair func.sum() with .group_by() to get a sum of the population by State and use the label() method to name the output.

# We can also create the func.sum() expression before using it in the select statement. We do it the same way we would inside the select statement and store it in a variable. Then we use that variable in the select statement where the func.sum() would normally be.

# Import func from sqlalchemy.
# Build an expression to calculate the sum of the values in the pop2008 field labeled as 'population'.
# Build a select statement to get the value of the state field and the sum of the values in pop2008.
# Group the statement by state using a .group_by() method.
# Execute stmt using the connection to get the count and store the results as results.
# Print the keys/column names of the results returned using results[0].keys()

# Import func
from sqlalchemy import func

# Build an expression to calculate the sum of pop2008 labeled as population
pop2008_sum = func.sum(census.columns.pop2008).label('population')

# Build a query to select the state and sum of pop2008: stmt
stmt = select([census.columns.state, pop2008_sum])

# Group stmt by state
stmt = stmt.group_by(census.columns.state)

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())

# DataFrame can take a SQLAlchemy ResultSet
# Make sure to set the DataFrame columns to the ResultSet keys

# SQLAlchemy ResultsProxy and Pandas Dataframes
# We can feed a ResultProxy directly into a pandas DataFrame, which is the workhorse of many Data Scientists in PythonLand. Jason demonstrated this in the video. In this exercise, you'll follow exactly the same approach to convert a ResultProxy into a DataFrame.

# Import pandas as pd.
# Create a DataFrame df using pd.DataFrame() on the ResultProxy results.
# Set the columns of the DataFrame df.columns to be the columns from the first result object results[0].keys().
# Print the DataFrame.

# import pandas
import pandas as pd

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the Dataframe
print(df)

# From SQLAlchemy results to a Graph
# We can also take advantage of pandas and Matplotlib to build figures of our data. Remember that data visualization is essential for both exploratory data analysis and communication of your data!

# Import matplotlib.pyplot as plt.
# Create a DataFrame df using pd.DataFrame() on the provided results.
# Set the columns of the DataFrame df.columns to be the columns from the first result object results[0].keys().
# Print the DataFrame df.
# Use the plot.bar() method on df to create a bar plot of the results.
# Display the plot with plt.show()

# Import pyplot as plt from matplotlib
import matplotlib.pyplot as plt

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set Column names
df.columns = results[0].keys()

# Print the DataFrame
print(df)

# Plot the DataFrame
df.plot.bar()
plt.show()

from sqlalchemy import case

stmt = select([
    func.sum(
        case([
            (census.columns.state == 'New York',
             census.columns.pop2008)
        ], else_=0))])

# Connecting to a MySQL Database
# Before you jump into the calculation exercises, let's begin by connecting to our database. Recall that in the last chapter you connected to a PostgreSQL database. Now, you'll connect to a MySQL database, for which many prefer to use the pymysql database driver, which, like psycopg2 for PostgreSQL, you have to install prior to use.

# This connection string is going to start with 'mysql+pymysql://', indicating which dialect and driver you're using to establish the connection. The dialect block is followed by the 'username:password' combo. Next, you specify the host and port with the following '@host:port/'. Finally, you wrap up the connection string with the 'database_name'.

# Now you'll practice connecting to a MySQL database: it will be the same census database that you have already been working with. One of the great things about SQLAlchemy is that, after connecting, it abstracts over the type of database it has connected to and you can write the same SQLAlchemy code, regardless!

# Import the create_engine function from the sqlalchemy library.
# Create an engine to the census database by concatenating the following strings and passing them to create_engine():
'mysql+pymysql://' (the dialect and driver).
'student:datacamp' (the username and password).
'@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/' (the host and port).
'census' (the database name).
# Use the .table_names() method on engine to print the table names.

