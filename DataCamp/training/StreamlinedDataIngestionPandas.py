# Get data from other flat files
# While CSVs are the most common kind of flat file, you will sometimes find files that use different delimiters. read_csv() can load all of these with the help of the sep keyword argument. By default, pandas assumes that the separator is a comma, which is why we do not need to specify sep for CSVs.
#
# The version of Vermont tax data here is a tab-separated values file (TSV), so you will need to use sep to pass in the correct delimiter when reading the file. Remember that tabs are represented as \t. Once the file has been loaded, the remaining code will create a chart of tax returns by income group.

# Import pandas with the alias pd
import pandas as pd

# Load TSV using the sep keyword argument to set delimiter
data = pd.read_csv('vt_tax_data_2016.tsv', sep='\t')

# Plot the total number of tax returns by income group
counts = data.groupby("agi_stub").N1.sum()
counts.plot.bar()
plt.show()

# Import a subset of columns
# The Vermont tax data contains 147 columns describing household composition, income sources, and taxes paid by ZIP code and income group. Most analyses don't need all these columns. In this exercise, you will create a data frame with fewer variables using read_csv()s usecols argument.
#
# Let's focus on household composition to see if there are differences by geography and income level. To do this, we'll need columns on income group, ZIP code, tax return filing status (e.g., single or married), and dependents. The data uses codes for variable names, so the specific columns needed are in the instructions.
#
# pandas has already been imported as pd.

# Create a list of columns to use: zipcode, agi_stub (income group), mars1 (number of single households), MARS2 (number of households filing as married), and NUMDEP (number of dependents).
# Create a data frame from vt_tax_data_2016.csv that uses only the selected columns.

# Create list of columns to use
cols = ['zipcode', 'agi_stub', 'mars1', 'MARS2', 'NUMDEP']

# Create data frame from csv using only selected columns
data = pd.read_csv("vt_tax_data_2016.csv", usecols=cols)

# View counts of dependents and tax returns by income level
print(data.groupby("agi_stub").sum())

# Import a file in chunks
# When working with large files, it can be easier to load and process the data in pieces. Let's practice this workflow on the Vermont tax data.
#
# The first 500 rows have been loaded as vt_data_first500. You'll get the next 500 rows. To do this, you'll have to employ several keyword arguments: nrows and skiprows to get the correct records, header to tell pandas the data does not have column names, and names to supply the missing column names. You'll also want to use the list() function to get column names from vt_data_first500 to reuse.
#
# pandas has been imported as pd.

# Use nrows and skiprows to create a data frame, vt_data_next500, containing the next 500 rows of the tax data.

# Create data frame of next 500 rows
vt_data_next500 = pd.read_csv("vt_tax_data_2016.csv",
                       		  skiprows = 500,
                       		  nrows = 500)

# View the data frame's head
print(vt_data_next500.head())

# Set the header argument so that pandas knows there is no header row.

# Name the columns in vt_data_next500 by using the appropriate keyword to pass in a list of vt_data_first500's columns.

# Create data frame of next 500 rows with labeled columns
vt_data_next500 = pd.read_csv("vt_tax_data_2016.csv",
                       		  nrows=500,
                       		  skiprows=500,
                       		  header=None,
                       		  names=list(vt_data_first500))

# View the Vermont data frames to confirm they're different
print(vt_data_first500.head())
print(vt_data_next500.head())

# dtype = {"zipcode": str}
# Object is the Pandas counterpart to Python strings

# error_bad_lines = False. This is to skup unparseable records

# warn_bad_lines = True. This is to see messages when records are skuped

# Specify data types
# When loading a flat file, pandas infers the best data type for each column. Sometimes its guesses are off, particularly for numbers that represent groups or qualities instead of quantities.
#
# Looking at the data dictionary for vt_tax_data_2016.csv, we can see two such columns. The agi_stub column contains numbers that correspond to income categories, and zipcode has 5-digit values that should be strings -- treating them as integers means we lose leading 0s, which are meaningful. Let's specify the correct data types with the dtype argument.
#
# pandas has been imported for you as pd.

# Load vt_tax_data_2016.csv with no arguments and view the data frame's dtypes attribute. Note the data types of zipcode and agi_stub.

# Load csv with no additional arguments
data = pd.read_csv("vt_tax_data_2016.csv")

# Print the data types
print(data.dtypes)

# Create a dictionary, data_types, specifying that agi_stub is 'category' data and zipcode is string data.
# Reload the CSV with the dtype argument and the dictionary to set the correct column data types.
# View the data frame's dtypes attribute.

# Create dict specifying data types for agi_stub and zipcode
data_types = {'agi_stub': 'category',
			  'zipcode': str}

# Load csv using dtype to set correct data types
data = pd.read_csv("vt_tax_data_2016.csv", dtype=data_types)

# Print data types of resulting frame
print(data.dtypes.head())

# Set custom NA values
# Part of data exploration and cleaning consists of checking for missing or NA values and deciding how to account for them. This is easier when missing values are treated as their own data type. and there are pandas functions that specifically target such NA values. pandas automatically treats some values as missing, but we can pass additional NA indicators with the na_values argument. Here, you'll do this to ensure that invalid ZIP codes in the Vermont tax data are coded as NA.
#
# pandas has been imported as pd.

# Create a dictionary, null_values, specifying that 0 in the zipcode column should be considered NA values.
# Load vt_tax_data_2016.csv, using the na_values argument and the dictionary to make sure invalid ZIP codes are treated as missing.

# Create dict specifying that 0s in zipcode are NA values
null_values = {"zipcode": 0}

# Load csv using na_values keyword argument
data = pd.read_csv("vt_tax_data_2016.csv",
                   na_values = null_values)

# View rows with NA ZIP codes
print(data[data.zipcode.isna()])

# Now that NA values are marked as such, it's possible to use NA-specific functions to do things like count missing values, as we did here, or drop records with missing values.

# Skip bad data
# In this exercise you'll use read_csv() parameters to handle files with bad data, like records with more values than columns. By default, trying to import such files triggers a specific error, pandas.io.common.CParserError.
#
# Some lines in the Vermont tax data here are corrupted. In order to load the good lines, we need to tell pandas to skip errors. We also want pandas to warn us when it skips a line so we know the scope of data issues.
#
# pandas has been imported as pd. The exercise code will try to read the file. If there is a pandas.io.common.CParserError, the code in the except block will run.

# Try to import the file vt_tax_data_2016_corrupt.csv without any keyword arguments.

try:
    # Import the CSV without any keyword arguments
    data = pd.read_csv('vt_tax_data_2016_corrupt.csv')

    # View first 5 records
    print(data.head())

except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")

# Import vt_tax_data_2016_corrupt.csv with the error_bad_lines parameter set to skip bad records.

try:
    # Import CSV with error_bad_lines set to skip bad records
    data = pd.read_csv("vt_tax_data_2016_corrupt.csv",
                       error_bad_lines=False)

    # View first 5 records
    print(data.head())

except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")

# Update the import with the warn_bad_lines parameter set to issue a warning whenever a bad record is skipped.

try:
    # Set warn_bad_lines to issue warnings about bad records
    data = pd.read_csv("vt_tax_data_2016_corrupt.csv",
                       error_bad_lines=False,
                       warn_bad_lines=True)

    # View first 5 records
    print(data.head())

except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")

# Get data from a spreadsheet
# In this exercise, you'll create a data frame from a "base case" Excel file: one with a single sheet of tabular data. The fcc_survey.xlsx file here has a sample of responses from FreeCodeCamp's annual New Developer Survey. This survey asks participants about their demographics, education, work and home life, plus questions about how they're learning to code. Let's load all of it.
#
# pandas has not been pre-loaded in this exercise, so you'll need to import it yourself before using read_excel() to load the spreadsheet.

# Load the pandas library as pd.
# Read in fcc_survey_simple.xlsx and assign it to the variable survey_responses.
# Print the first few records of survey_responses.

# Load pandas as pd
import pandas as pd

# Read spreadsheet and assign it to survey_responses
survey_responses = pd.read_excel('fcc_survey.xlsx')

# View the head of the data frame
print(survey_responses.head())

# Load a portion of a spreadsheet
# Spreadsheets meant to be read by people often have multiple tables, e.g., a small business might keep an inventory spreadsheet with tables for different product types on a single sheet. Even tabular data may have header rows of metadata, like the New Developer Survey data here. While the metadata is useful, we don't want it in a data frame. You'll use read_excel()'s skiprows keyword to get just the data. You'll also use usecols to focus on columns AD and AW through BA, about future job goals.
#
# pandas has been imported as pd.

# Create a string, col_string, specifying that pandas should load only columns AD and the range AW through BA.
# Load fcc_survey_headers.xlsx', setting skiprows and usecols to skip the first two rows of metadata and get only the columns in col_string.
# View the selected column names in the resulting data frame.

# Create string of lettered columns to load
col_string = "AD, AW:BA"

# Load data with skiprows and usecols set
survey_responses = pd.read_excel("fcc_survey_headers.xlsx",
                        skiprows = 2,
                        usecols = col_string)

# View the names of the columns selected
print(survey_responses.columns)

# Check equality of dfs
print(df1.equals(df2))

# Loading all sheets
# We specify sheet_name=None to read_excel and we get back an ordered dict

for key, value in survey_daya.items():
    print(key, type(value))

all_responses = pd.DataFrame()

for sheet_name, frame in survey_responses.items():
    frame["Year"] = sheet_name

    all_responses = all_responses.append(frame)

# Select a single sheet
# An Excel workbook may contain multiple sheets of related data. The New Developer Survey response workbook has sheets for different years. Because read_excel() loads only the first sheet by default, you've already gotten survey responses for 2016. Now, you'll create a data frame of 2017 responses using read_excel()'s sheet_name argument in a couple different ways.
#
# pandas has been imported as pd.

# Create a data frame from the second workbook sheet by passing the sheet's position to sheet_name.

# Create df from second worksheet by referencing its position
responses_2017 = pd.read_excel("fcc_survey.xlsx",
                               sheet_name = 1)

# Graph where people would like to get a developer job
job_prefs = responses_2017.groupby("JobPref").JobPref.count()
job_prefs.plot.barh()
plt.show()

# Create a data frame from the 2017 sheet by providing the sheet's name to `read_excel()``.

# Create df from second worksheet by referencing its position
responses_2017 = pd.read_excel("fcc_survey.xlsx",
                               sheet_name = '2017')

# Graph where people would like to get a developer job
job_prefs = responses_2017.groupby("JobPref").JobPref.count()
job_prefs.plot.barh()
plt.show()

# Select multiple sheets
# So far, you've read Excel files one sheet at a time, which lets you you customize import arguments for each sheet. But if an Excel file has some sheets that you want loaded with the same parameters, you can get them in one go by passing a list of their names or indices to read_excel()'s sheet_name keyword. To get them all, pass None. You'll practice both methods to get data from fcc_survey.xlsx, which has multiple sheets of similarly-formatted data.
#
# pandas has been loaded as pd.

# Load both the 2016 and 2017 sheets by name with a list and one call to read_excel().

# Load both the 2016 and 2017 sheets by name
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=['2016', '2017'])

# View the data type of all_survey_data
print(type(all_survey_data))

# Load the 2016 sheet by its position (0) and 2017 by name. Note the sheet names in the result.

# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name = [0, '2017'])

# View the sheet names in all_survey_data
print(all_survey_data.keys())

# Load all sheets in the Excel file without listing them all.
# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=None)

# View the sheet names in all_survey_data
print(all_survey_data.keys())

# Work with multiple spreadsheets
# Workbooks meant primarily for human readers, not machines, may store data about a single subject across multiple sheets. For example, a file may have a different sheet of transactions for each region or year in which a business operated.
#
# The FreeCodeCamp New Developer Survey file is set up similarly, with samples of responses from different years in different sheets. Your task here is to compile them in one data frame for analysis.
#
# pandas has been imported as pd. All sheets have been read into the ordered dictionary responses.

# Create an empty data frame, all_responses.
# Set up a for loop to iterate through the values in responses
# Append each data frame to all_responses and reassign the result to the same variable name.

# Create an empty data frame
all_responses = pd.DataFrame()

# Set up for loop to iterate through values in responses
for df in responses.values():
  # Print the number of rows being added
  print("Adding {} rows".format(df.shape[0]))
  # Append df to all_responses, assign result
  all_responses = all_responses.append(df)

# Graph employment statuses in sample
counts = all_responses.groupby("EmploymentStatus").EmploymentStatus.count()
counts.plot.barh()
plt.show()

# Good work! You compiled similar spreadsheets into one dataset. This method works well when you know your spreadsheets use the same column names. If they don't, you can end up with lots of NA values where column names don't align.

# Set Boolean columns
# Datasets may have columns that are most accurately modeled as Boolean values. However, pandas usually loads these as floats by default, since defaulting to Booleans may have undesired effects like turning NA values into Trues.
#
# fcc_survey_subset.xlsx contains a string ID column and several True/False columns indicating financial stressors. You'll evaluate which non-ID columns have no NA values and therefore can be set as Boolean, then tell read_excel() to load them as such with the dtype argument.
#
# pandas is loaded as pd.

# Count NA values in each column of survey_data with isna() and sum(). Note which columns besides ID.x, if any, have zero NAs.

# Load the data
survey_data = pd.read_excel("fcc_survey_subset.xlsx")

# Count NA values in each column
print(survey_data.isna().sum())

# Set dtype to load appropriate column(s) as Boolean data
survey_data = pd.read_excel("fcc_survey_subset.xlsx",
                            dtype = {"HasDebt": bool})

# View financial burdens by Boolean group
print(survey_data.groupby("HasDebt").sum())

# Great work! Modeling True/False data as Booleans can streamline some data manipulation functions and keep spurious summary statistics, like quartile values, from being calculated. If you want to make a column with NA values Boolean, you can load the data, impute missing values, then re-cast the column as Boolean.

# Set custom true/false values
# In Boolean columns, pandas automatically recognizes certain values, like "TRUE" and 1, as True, and others, like "FALSE" and 0, as False. Some datasets, like survey data, can use unrecognized values, such as "Yes" and "No".
#
# For practice purposes, some Boolean columns in the New Developer Survey have been coded this way. You'll make sure they're properly interpreted with the help of the true_values and false_values arguments.
#
# pandas is loaded as pd. You can assume the columns you are working with have no missing values.

# Load the Excel file, specifying "Yes" as a true value and "No" as a false value.

# Load file with Yes as a True value and No as a False value
survey_subset = pd.read_excel("fcc_survey_yn_data.xlsx",
                              dtype={"HasDebt": bool,
                              "AttendedBootCampYesNo": bool},
                              true_values = ["Yes"],
                              false_values = ["No"])

# View the data
print(survey_subset.head())

# Use pd.to_datetime() after loading data if parse_dates won't work

# to_datetime() argunents:
# a. Df and column to convert
# b. format: string representation of datetime format

# Parse simple dates
# pandas does not infer that columns contain datetime data; it interprets them as object or string data unless told otherwise. Correctly modeling datetimes is easy when they are in a standard format -- we can use the parse_dates argument to tell read_excel() to read columns as datetime data.
#
# The New Developer Survey responses contain some columns with easy-to-parse timestamps. In this exercise, you'll make sure they're the right data type.
#
# pandas has been loaded as pd.

# Load fcc_survey.xlsx, making sure that the Part1StartTime column is parsed as datetime data.

# View the first few values of the survey_data.Part1StartTime to make sure it contains datetimes.

# Get datetimes from multiple columns
# Sometimes, datetime data is split across columns. A dataset might have a date and a time column, or a date may be split into year, month, and day columns.
#
# A column in this version of the survey data has been split so that dates are in one column, Part2StartDate, and times are in another, Part2StartTime. Your task is to use read_excel()'s parse_dates argument to combine them into one datetime column with a new name.
#
# pandas has been imported as pd.

# Create a dictionary indicating that the new column Part2Start should consist of Part2StartDate and Part2StartTime.
# Load the survey response file, using parse_dates and the dictionary to create a new Part2Start column.
# View summary statistics about the new Part2Start column with the describe() method.

# Create dict of columns to combine into new datetime column
datetime_cols = {"Part2Start": ["Part2StartDate", "Part2StartTime"]}


# Load file, creating a new Part2Start column
survey_data = pd.read_excel("fcc_survey_dts.xlsx",
                            parse_dates = datetime_cols)

# View summary statistics about Part2Start
print(survey_data.Part2Start.describe())

# Well done! Note that the keys in a dictionary passed to parse_dates cannot be names of columns already in the data frame. Also, when combining columns to parse, their order in the list does not matter.

# Parse non-standard date formats
# So far, you've parsed dates that pandas could interpret automatically. But if a date is in a non-standard format, like 19991231 for December 31, 1999, it can't be parsed at the import stage. Instead, use pd.to_datetime() to convert strings to dates after import.
#
# The New Developer Survey data has been loaded as survey_data but contains an unparsed datetime field. We'll use to_datetime() to convert it, passing in the column to convert and a string representing the date format used.
#
# For more on date format codes, see this reference. Some common codes are year (%Y), month (%m), day (%d), hour (%H), minute (%M), and second (%S).
#
# pandas has been imported as pd.

# Create a database engine using SQLAlchemy's create_engine(): makes an engine to handle db connections
from sqlalchemy import create_engine

# Connect to a database
# In order to get data from a database with pandas, you first need to be able to connect to one. In this exercise, you'll practice creating a database engine to manage connections to a database, data.db. To do this, you'll use sqlalchemy's create_engine() function.
#
# create_engine() needs a string URL to the database. For SQLite databases, that string consists of "sqlite:///", then the database file name.

# Use create_engine() to make a database engine for data.db.
# Run the last line of code to show the names of the tables in the database.

# Import sqlalchemy's create_engine() function
from sqlalchemy import create_engine

# Create the database engine
engine = create_engine("sqlite:///data.db")

# View the tables in the database
print(engine.table_names())

# Load entire tables
# In the last exercise, you saw that data.db has two tables. weather has historical weather data for New York City. hpd311calls is a subset of call records made to the city's 311 help line about housing issues.
#
# In this exercise, you'll use the read_sql() function in pandas to load both tables. read_sql() accepts a string of either a SQL query to run, or a table to load. It also needs a way to connect to the database, like the engine in the provided code.

# Use read_sql() to load the hpd311calls table by name, without any SQL.

# Load libraries
import pandas as pd
from sqlalchemy import create_engine

# Create the database engine
engine = create_engine('sqlite:///data.db')

# Load hpd311calls without any SQL
hpd_calls = pd.read_sql("hpd311calls", engine)

# View the first few rows of data
print(hpd_calls.head())

# Use read_sql() and a SELECT * ... SQL query to load the entire weather table.
# Create the database engine
engine = create_engine("sqlite:///data.db")

# Create a SQL query to load the entire weather table
query = """
SELECT * 
  FROM weather;
"""

# Load weather with the SQL query
weather = pd.read_sql(query, engine)

# View the first few rows of data
print(weather.head())

# Wrapping the string in triple quotes lets us split it between multiple lines so it's easier to read

# Selecting columns with SQL
# Datasets can contain columns that are not required for an analysis, like the weather table in data.db does. Some, such as elevation, are redundant, since all observations occurred at the same place, while others contain variables we are not interested in. After making a database engine, you'll write a query to SELECT only the date and temperature columns, and pass both to read_sql() to make a data frame of high and low temperature readings.
#
# pandas has been loaded as pd, and create_engine() has been imported from sqlalchemy.

# Create a database engine for data.db.
# Write a SQL query that SELECTs the date, tmax, and tmin columns from the weather table.
# Make a data frame by passing the query and engine to read_sql() and assign the resulting data frame to temperatures.

# Create database engine for data.db
engine = create_engine("sqlite:///data.db")

# Write query to get date, tmax, and tmin from weather
query = """
SELECT date, 
       tmax, 
       tmin
  FROM weather;
"""

# Make a data frame by passing query and engine to read_sql()
temperatures = pd.read_sql(query, engine)

# View the resulting data frame
print(temperatures)

# Selecting columns is useful when you only want a few columns from a table. If you want most of the columns, it may be easier to load them all and then use pandas to drop unwanted columns.

# Selecting rows
# SQL WHERE clauses return records whose values meet the given criteria. Passing such a query to read_sql() results in a data frame loaded with only records we are interested in, so there is less filtering to do later on.
#
# The hpd311calls table in data.db has data on calls about various housing issues, from maintenance problems to information requests. In this exercise, you'll use SQL to focus on calls about safety.
#
# pandas has been loaded as pd, and a database engine, engine, has been created for data.db

# Create a query that selects all columns of records in hpd311calls that have 'SAFETY' as their complaint_type.
# Use read_sql() to query the database and assign the result to the variable safety_calls.
# Run the last section of code to create a graph of safety call counts in each borough.

# Create query to get hpd311calls records about safety
query = """
SELECT *
FROM hpd311calls
WHERE complaint_type = 'SAFETY';
"""

# Query the database and assign result to safety_calls
safety_calls = pd.read_sql(query, engine)

# Graph the number of safety calls by borough
call_counts = safety_calls.groupby('borough').unique_key.count()
call_counts.plot.barh()
plt.show()

# Filtering on multiple conditions
# So far, you've selectively imported records that met a single condition, but it's also common to filter datasets on multiple criteria. In this exercise, you'll do just that.
#
# The weather table contains daily high and low temperatures and precipitation amounts for New York City. Let's focus on inclement weather, where there was either an inch or more of snow or the high was at or below freezing (32° Fahrenheit). To do this, you'll need to build a query that uses the OR operator to look at values in both columns.
#
# pandas is loaded as pd, and a database engine, engine, has been created.

# Create a query that selects records in weather where tmax is less than or equal to 32 degrees OR snow is greater than or equal to 1 inch.
# Use read_sql() to query the database and assign the result to the variable wintry_days.
# View summary statistics with the describe() method to make sure all records in the data frame meet the given criteria.

# Create query for records with max temps <= 32 or snow >= 1
query = """
SELECT *
  FROM weather
  WHERE tmax <= 32
  OR snow >= 1;
"""

# Query database and assign result to wintry_days
wintry_days = pd.read_sql(query, engine)

# View summary stats about the temperatures
print(wintry_days.describe())

# Nice work. SELECT statements can use multiple AND and OR operators to filter data. Like arithmetic, you can control the order of operations with parentheses.

# Get number of unique values in a column
SELECT COUNT(DISTINCT [column_names]) FROM [table_name]

# Getting distinct values
# Sometimes an analysis doesn't need every record, but rather unique values in one or more columns. Duplicate values can be removed after loading data into a data frame, but it can also be done at import with SQL's DISTINCT keyword.
#
# Since hpd311calls contains data about housing issues, we would expect most records to have a borough listed. Let's test this assumption by querying unique complaint_type/borough combinations.
#
# pandas has been imported as pd, and the database engine has been created as engine.

# Create a query that gets DISTINCT values for borough and complaint_type from hpd311calls.
# Use read_sql() to load the results of the query to a data frame, issues_and_boros.
# Print the data frame to check if the assumption that all issues besides literature requests are appear with boroughs listed.

# SELECT DISTINCT does not take column names in parentheses.
# DISTINCT only needs to be specified once -- the query will automatically get unique combinations of all columns named after the keyword.
# Be sure to print the entire data frame, not just the head.

# Create query for unique combinations of borough and complaint_type
query = """
SELECT DISTINCT borough, 
       complaint_type
  FROM hpd311calls;
"""

# Load results of query to a data frame
issues_and_boros = pd.read_sql(query, engine)

# Check assumption about issues and boroughs
print(issues_and_boros)

# Counting in groups
# In previous exercises, you pulled data from tables, then summarized the resulting data frames in pandas to create graphs. By using COUNT and GROUP BY in a SQL query, we can pull those summary figures from the database directly.
#
# The hpd311calls table has a column, complaint_type, that categorizes call records by issue, such as heating or plumbing. In order to graph call volumes by issue, you'll write a SQL query that COUNTs records by complaint type.
#
# pandas has been imported as pd, and the database engine for data.db has been created as engine.

# Create a SQL query that gets complaint_type and counts of all records from hpd311calls, grouped by complaint_type.
# Create a data frame with read_sql() of call counts by issue, calls_by_issue.
# Run the last section of code to graph the number of calls for each housing issue.

# Create query to get call counts by complaint_type
query = """
SELECT complaint_type,
     COUNT(*)
  FROM hpd311calls
GROUP BY complaint_type;
"""

# Create data frame of call counts by issue
calls_by_issue = pd.read_sql(query, engine)

# Graph the number of calls for each housing issue
calls_by_issue.plot.barh(x="complaint_type")
plt.show()

# Working with aggregate functions
# If a table contains data with higher granularity than is needed for an analysis, it can make sense to summarize the data with SQL aggregate functions before importing it. For example, if you have data of flood event counts by month but precipitation data by day, you may decide to SUM precipitation by month.
#
# The weather table contains daily readings for four months. In this exercise, you'll practice summarizing weather by month with the MAX, MIN, and SUM functions.
#
# pandas has been loaded as pd, and a database engine, engine, has been created.

# Create a query to pass to read_sql() that will get months and the MAX value of tmax by monthfrom weather.

# Create a query to get month and max tmax by month
query = """
SELECT month, 
       MAX(tmax)
  FROM weather 
  GROUP BY month;"""

# Get data frame of monthly weather stats
weather_by_month = pd.read_sql(query, engine)

# View weather stats by month
print(weather_by_month)

# Modify the query to also get the MIN tmin value for each month.

# Create a query to get month, max tmax, and min tmin by month
query = """
SELECT month, 
	   MAX(tmax), 
       MIN(tmin)
  FROM weather 
 GROUP BY month;
"""

# Get data frame of monthly weather stats
weather_by_month = pd.read_sql(query, engine)

# View weather stats by month
print(weather_by_month)

# Modify the query to also get the total precipitation (prcp) for each month.

# Create query to get temperature and precipitation by month
query = """
SELECT month, 
        MAX(tmax), 
        MIN(tmin),
        SUM(prcp)
  FROM weather 
 GROUP BY month;
"""

# Get data frame of monthly weather stats
weather_by_month = pd.read_sql(query, engine)

# View weather stats by month
print(weather_by_month)

# Well done! Aggregate functions can be a useful way to summarize large datasets. Different database management systems even have SQL functions for statistics like standard deviation and percentiles, though these are non-standard and vendor-specific.

# Joining tables
# Tables in in relational databases usually have key columns of unique record identifiers. This lets us build pipelines that combine tables using SQL's JOIN operation, instead of having to combine data after importing it.
#
# The records in hpd311calls often concern issues, like leaks or heating problems, that are exacerbated by weather conditions. In this exercise, you'll join weather data to call records along their common date columns to get everything in one data frame. You can assume these columns are the same data type.
#
# pandas is loaded as pd, and the database engine, engine, has been created.

# Complete the query to join weather to hpd311calls by their date and created_date columns, respectively.
# Query the database and assign the resulting data frame to calls_with_weather.
# Print the first few rows of calls_with_weather to confirm all columns were joined.

# Query to join weather to call records by date columns
query = """
SELECT * 
  FROM hpd311calls
       JOIN weather 
       ON hpd311calls.created_date = weather.date;
"""

# Create data frame of joined tables
calls_with_weather = pd.read_sql(query, engine)

# View the data frame to make sure all columns were joined
print(calls_with_weather.head())

# Joining and filtering
# Just as you might not always want all the data in a single table, you might not want all columns and rows that result from a JOIN. In this exercise, you'll use SQL to refine a data import.
#
# Weather exacerbates some housing problems more than others. Your task is to focus on water leak reports in hpd311calls and assemble a dataset that includes the day's precipitation levels from weather to see if there is any relationship between the two. The provided SQL gets all columns in hpd311calls, but you'll need to modify it to get the necessary weather column and filter rows with a WHERE clause.
#
# pandas is loaded as pd, and the database engine, engine, has been created.

# Complete query to get the prcp column in weather and join weather to hpd311calls on their date and created_date columns, respectively.

# Use read_sql() to load the results of the query into the leak_calls data frame.

# Query to get hpd311calls and precipitation values
query = """
SELECT hpd311calls.*, weather.prcp
  FROM hpd311calls
  JOIN weather
  ON hpd311calls.created_date = weather.date"""

# Load query results into the leak_calls data frame
leak_calls = pd.read_sql(query, engine)

# View the data frame
print(leak_calls.head())

# Modify query to get only rows that have 'WATER LEAK' as their complaint_type
# Query to get water leak calls and daily precipitation
query = """
SELECT hpd311calls.*, weather.prcp
  FROM hpd311calls
  JOIN weather
    ON hpd311calls.created_date = weather.date
  WHERE hpd311calls.complaint_type = 'WATER LEAK';
"""

# Load query results into the leak_calls data frame
leak_calls = pd.read_sql(query, engine)

# View the data frame
print(leak_calls.head())

# Joining, filtering, and aggregating
# In this exercise, you'll use what you've learned to assemble a dataset to investigate how the number of heating complaints to New York City's 311 line varies with temperature.
#
# In addition to the hpd311calls table, data.db has a weather table with daily high and low temperature readings for NYC. We want to get each day's count of heat/hot water calls with temperatures joined in. This can be done in one query, which we'll build in parts.
#
# pandas has been imported as pd, and the database engine has been created as engine.

# Write a query to get created_date and counts of records whose complaint_type is HEAT/HOT WATER from hpd311calls by date.
# Create a data frame,df, containing the results of the query.

# Query to get heat/hot water call counts by created_date
query = """
SELECT created_date, 
       COUNT(*)
  FROM hpd311calls 
        WHERE complaint_type = 'HEAT/HOT WATER'
        GROUP BY created_date;"""

# Query database and save results as df
df = pd.read_sql(query, engine)

# View first 5 records
print(df.head())

# Modify the query to join tmax and tmin from the weather table. The tables should be joined on created_date in hpd311calls and date in weather

# Modify query to join tmax and tmin from weather by date
query = """
SELECT created_date, 
	   COUNT(*), 
       weather.tmax,
       weather.tmin
  FROM hpd311calls 
       JOIN weather
       ON hpd311calls.created_date = weather.date
 WHERE complaint_type = 'HEAT/HOT WATER' 
 GROUP BY created_date;"""

# Query database and save results as df
df = pd.read_sql(query, engine)

# View first 5 records
print(df.head())

# Load JSON data
# Many open data portals make available JSONs datasets that are particularly easy to parse. They can be accessed directly via URL. Each object is a record, all objects have the same set of attributes, and none of the values are nested objects that themselves need to be parsed.
#
# The New York City Department of Homeless Services Daily Report is such a dataset, containing years' worth of homeless shelter population counts. You can view it in the console before loading it to a data frame with pandas's read_json() function.

# Get a sense of the contents of dhs_daily_report.json, which are printed in the console.
# Load pandas as pd.
# Use read_json() to load dhs_daily_report.json to a data frame, pop_in_shelters.
# View summary statistics about pop_in_shelters with the data frame's describe() method.

# Load pandas as pd
import pandas as pd

# Load the daily report to a data frame
pop_in_shelters = pd.read_json('dhs_daily_report.json')

# View summary stats about pop_in_shelters
print(pop_in_shelters.describe())

# Work with JSON orientations
# JSON isn't a tabular format, so pandas makes assumptions about its orientation when loading data. Most JSON data you encounter will be in orientations that pandas can automatically transform into a data frame.
#
# Sometimes, like in this modified version of the Department of Homeless Services Daily Report, data is oriented differently. To reduce the file size, it has been split formatted. You'll see what happens when you try to load it normally versus with the orient keyword argument. The try/except block will alert you if there are errors loading the data.
#
# pandas has been loaded as pd.

# Try loading dhs_report_reformatted.json without any keyword arguments.

try:
    # Load the JSON without keyword arguments
    df = pd.read_json('dhs_report_reformatted.json')

    # Plot total population in shelters over time
    df["date_of_census"] = pd.to_datetime(df["date_of_census"])
    df.plot(x="date_of_census",
            y="total_individuals_in_shelter")
    plt.show()

except ValueError:
    print("pandas could not parse the JSON.")

try:
    # Load the JSON with orient specified
    df = pd.read_json("dhs_report_reformatted.json",
                      orient='split')

    # Plot total population in shelters over time
    df["date_of_census"] = pd.to_datetime(df["date_of_census"])
    df.plot(x="date_of_census",
            y="total_individuals_in_shelter")
    plt.show()

except ValueError:
    print("pandas could not parse the JSON.")

# Get data from an API
# In this exercise, you'll use requests.get() to query the Yelp Business Search API for cafes in New York City. requests.get() needs a URL to get data from. The Yelp API also needs search parameters and authorization headers passed to the params and headers keyword arguments, respectively.
#
# You'll need to extract the data from the response with its json() method, and pass it to pandas's DataFrame() function to make a data frame. Note that the necessary data is under the dictionary key "businesses".
#
# pandas (as pd) and requests have been loaded. Authorization data is in the dictionary headers, and the needed API parameters are stored as parameters.

# Get data about New York City cafes from the Yelp API (api_url) with requests.get() and the params and header dictionaries.
# Extract the JSON data from the response with its json() method, and assign it to data.
# Load the cafe listings to the data frame cafes with pandas's DataFrame() function. The listings are under the "businesses" key in data.
# Print the data frame's dtypes to see what information you're getting.

api_url = "https://api.yelp.com/v3/businesses/search"

# Get data about NYC cafes from the Yelp API
response = requests.get(api_url,
                headers=headers,
                params=parameters)

# Extract JSON data from the response
data = response.json()

# Load data to a data frame
cafes = pd.DataFrame(data["businesses"])

# View the data's dtypes
print(cafes.dtypes)

# Set API parameters
# Formatting parameters to get the data you need is an integral part of working with APIs. These parameters can be passed to the get() function's params keyword argument as a dictionary.
#
# The Yelp API requires the location parameter be set. It also lets users supply a term to search for. You'll use these parameters to get data about cafes in NYC, then process the result to create a data frame.
#
# pandas (as pd) and requests have been loaded. The API endpoint is stored in the variable api_url. Authorization data is stored in the dictionary headers

# Create a dictionary, parameters, with the term and location parameters set to search for "cafe"s in "NYC".
# Query the Yelp API (api_url) with requests's get() function and the headers and params keyword arguments set. Save the result as response.
# Extract the JSON data from response with the appropriate method. Save the result as data.
# Load the "businesses" values in data to the data frame cafes and print the head.

# Create dictionary to query API for cafes in NYC
parameters = {"term": "cafe",
          	  "location": "NYC"}

# Query the Yelp API with headers and params set
response = requests.get(api_url,
                headers=headers,
                params=parameters)

# Extract JSON data from response
data = response.json()

# Load "businesses" values to a data frame and print head
cafes = pd.DataFrame(data["businesses"])
print(cafes.head())

# Set request headers
# Many APIs require users provide an API key, obtained by registering for the service. Keys typically are passed in the request header, rather than as parameters.
#
# The Yelp API documentation says "To authenticate API calls with the API Key, set the Authorization HTTP header value as Bearer API_KEY."
#
# You'll set up a dictionary to pass this information to get(), call the API for the highest-rated cafes in NYC, and parse the response.
#
# pandas (as pd) and requests have been loaded. The API endpoint is stored as api_url, and the key is api_key. Parameters are in the dictionary params

# Create a dictionary, headers, that passes the formatted key string to the "Authorization" header value.
# Query the Yelp API (api_url) with get() and the necessary headers and parameters. Save the result as response.
# Extract the JSON data from response. Save the result as data.
# Load the "businesses" values in data to the data frame cafes and print the names column.

# Create dictionary that passes Authorization and key string
headers = {"Authorization": "Bearer {}".format(api_key)}

# Query the Yelp API with headers and params set
response = requests.get(api_url, headers=headers, params=parameters)

# Extract JSON data from response
data = response.json()

# Load "businesses" values to a data frame and print names
cafes = pd.DataFrame(data["businesses"])
print(cafes.name)

# pandas.io.json submodule contains tools for reading and writing JSON. Needs its own import statement
# We'll use its json normalize function to flatten nested data

from pandas.io.json import json_normalize

# Flatten nested JSONs
# A feature of JSON data is that it can be nested: an attribute's value can consist of attribute-value pairs. This nested data is more useful unpacked, or flattened, into its own data frame columns. The pandas.io.json submodule has a function, json_normalize(), that does exactly this.
#
# The Yelp API response data is nested. Your job is to flatten out the next level of data in the coordinates and location columns.
#
# pandas (as pd) and requests have been imported. The results of the API call are stored as response.

# Load the json_normalize() function from pandas' io.json submodule.
# Isolate the JSON data from response and assign it to data.
# Use json_normalize() to flatten and load the businesses data to a data frame, cafes. Set the sep argument to use underscores (_), rather than periods.
# Print the data head.

# Load json_normalize()
from pandas.io.json import json_normalize

# Isolate the JSON data from the API response
data = response.json()

# Flatten business data into a data frame, replace separator
cafes = json_normalize(data["businesses"],
             sep="_")

# View data
print(cafes.head())

# Handle deeply nested data
# Last exercise, you flattened data nested down one level. Here, you'll unpack more deeply nested data.
#
# The categories attribute in the Yelp API response contains lists of objects. To flatten this data, you'll employ json_normalize() arguments to specify the path to categories and pick other attributes to include in the data frame. You should also change the separator to facilitate column selection and prefix the other attributes to prevent column name collisions. We'll work through this in steps.
#
# pandas (as pd) and json_normalize() have been imported. JSON-formatted Yelp data on cafes in NYC is stored as data

# Use json_normalize() to flatten records under the businesses key in data, setting underscores (_) as separators.

# Flatten businesses records and set underscore separators
flat_cafes = json_normalize(data["businesses"],
                  sep="_")

# View the data
print(flat_cafes.head())

# Specify the record_path to the categories data.

# Specify record path to get categories data
flat_cafes = json_normalize(data["businesses"],
                            sep="_",
                    		record_path = "categories")

# View the data
print(flat_cafes.head())

# Set the meta keyword argument to get business name, alias, rating, and the attributes nested under coordinates: latitude and longitude.
# Add "biz_" as a meta_prefix to prevent duplicate column names.

# Load other business attributes and set meta prefix
flat_cafes = json_normalize(data["businesses"],
                            sep="_",
                    		record_path="categories",
                    		meta=["name",
                                  "alias",
                                  "rating",
                          		  ["coordinates", "latitude"],
                          		  ["coordinates", "longitude"]],
                    		meta_prefix="biz_")


# View the data
print(flat_cafes.head())

# Great job! Naming meta columns can get tedious for datasets with many attributes, and code is susceptible to breaking if column names or nesting levels change. In such cases, you may have to write a custom function and employ techniques like recursion to handle the data.

# append takes the dataframe to add on as an argument
# Set ignore_index to True to renumber rows (rather than 2 row 0s', 2 row 1s etc

# APIs commonly limit the nr of records returned in a single call to manage resource usage
# Set the offset parameter to get the next 20 etc

params["offset"] = 20

# merge is both a pandas function and a df method

# df.merge() argumentst
# second df to merge
# columns to merge on
# on if names are the same in both dfs
# left_on and right_on if key names differ

# Append data frames
# In this exercise, you’ll practice appending records by creating a dataset of the 100 highest-rated cafes in New York City according to Yelp.
#
# APIs often limit the amount of data returned, since sending large datasets can be time- and resource-intensive. The Yelp Business Search API limits the results returned in a call to 50 records. However, the offset parameter lets a user retrieve results starting after a specified number. By modifying the offset, we can get results 1-50 in one call and 51-100 in another. Then, we can append the data frames.
#
# pandas (as pd), requests, and json_normalize() have been imported. The 50 top-rated cafes are already in a data frame, top_50_cafes.

# Add an offset parameter to params so that the Yelp API call will get cafes 51-100.
# Append the results of the API call to top_50_cafes, setting ignore_index so rows will be renumbered.
# Print the shape of the resulting data frame, cafes, to confirm there are 100 records.

# Add an offset parameter to get cafes 51-100
params = {"term": "cafe",
          "location": "NYC",
          "sort_by": "rating",
          "limit": 50,
          "offset": 50}

result = requests.get(api_url, headers=headers, params=params)
next_50_cafes = json_normalize(result.json()["businesses"])

# Append the results, setting ignore_index to renumber rows
cafes = top_50_cafes.append(next_50_cafes, ignore_index=True)

# Print shape of cafes
print(cafes.shape)

# Nice work! If you were putting multiple data frames together, one option would be to start with an empty data frame and use a for or while loop to append additional ones.

# Merge the crosswalk data frame into cafes on zipcode and location_zip_code, respectively. Assign the result to cafes_with_pumas.
# Merge pop_data into the cafes_with_pumas on the puma field in both. Assign the result to cafes_with_pop

# Merge the crosswalk data into cafes on zip code fields
cafes_with_pumas = cafes.merge(crosswalk, left_on="location_zip_code", right_on="zipcode")



# Merge in population data on puma field
cafes_with_pop = cafes_with_pumas.merge(pop_data, on="puma")

# View the data
print(cafes_with_pop.head())

# Congratulations! You've built a pretty sophisticated pipeline that translates geographies to link data from multiple sources. While postal codes are a commonly used areal unit, there are often more meaningful ways to group spatial data, such as by neighborhood here.

