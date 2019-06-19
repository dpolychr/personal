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

