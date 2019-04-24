import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/dimitrispolychronopoulos/Documents/GEL_Tasks/FamSelectionOutput/RD37_fam_selection_output_2018_09_05_234455.csv')
fig, ax = plt.subplots()
ax.hist(df['N_of_Family_Participants'])

# This is with pandas
df['N_of_Family_Participants'].plot.hist()

# With seaborn; Note that this is actual a Gaussian kernel density estimate (KDE)
import seaborn as sns
sns.distplot(df['N_of_Family_Participants'], kde=False, bins=10)

# import all modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the DataFrame
df = pd.read_csv(grant_file)

# Comparing a histogram and distplot
# The pandas library supports simple plotting of data, which is very convenient when data is already likely to be in a pandas DataFrame.
#
# Seaborn generally does more statistical analysis on data and can provide more sophisticated insight into the data. In this exercise, we will compare a pandas histogram vs the seaborn distplot.

# Use the pandas' plot.hist() function to plot a histogram of the Award_Amount column.

# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()

# Clear out the pandas histogram
plt.clf()

# Use Seaborn's distplot() function to plot a distribution plot of the same column.

# Display a Seaborn distplot
sns.distplot(df['Award_Amount'])
plt.show()

# Clear the distplot
plt.clf()

# Alternative data distributions
# A rug plot is an alternative way to view the ditribution of data
# A kde curve and rug plot can be combined

# Plot a histogram
# The distplot() function will return a Kernel Density Estimate (KDE) by default. The KDE helps to smooth the distribution and is a useful way to look at the data. However, Seaborn can also support the more standard histogram approach if that is more meaningful for your analysis.

# Create a distplot for the data and disable the KDE.
# Explicitly pass in the number 20 for the number of bins in the histogram.
# Display the plot using plt.show()

# Create a distplot
sns.distplot(df['Award_Amount'],
             kde=False,
             bins=20)

# Display the plot
plt.show()

# Rug plot and kde shading
# Now that you understand some function arguments for distplot(), we can continue further refining the output. This process of creating a visualization and updating it in an incremental fashion is a useful and common approach to look at data from multiple perspectives.
#
# Seaborn excels at making this process simple.

# Create a distplot of the Award_Amount column in the df.
# Configure it to show a shaded kde (using the kde_kws dictionary).
# Add a rug plot above the x axis.
# Display the plot.

# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})

# Plot the results
plt.show()

# The regplot function generates a scatter plot with a regression line
# Usage is similar to the distplot
sns.regplot()

# Create a regression plot
# For this set of exercises, we will be looking at FiveThirtyEight's data on which US State has the worst drivers. The data set includes summary level information about fatal accidents as well as insurance premiums for each state as of 2010.
#
# In this exercise, we will look at the difference between the regression plotting functions.

# Plotting multiple variables
# Since we are using lmplot() now, we can look at the more complex interactions of data. This data set includes geographic information by state and area. It might be interesting to see if there is a difference in relationships based on the Region of the country.

# Use lmplot() to look at the relationship between insurance_losses and premiums.
# Plot a regression line for each Region of the country.

# Create a regression plot using hue
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           hue="Region")

# Show the results
plt.show()

# Facetting multiple regressions
# lmplot() allows us to facet the data across multiple rows and columns. In the previous plot, the multiple lines were difficult to read in one plot. We can try creating multiple plots by Region to see if that is a more useful visualization.

# Use lmplot() to look at the relationship between insurance_losses and premiums.
# Create a plot for each Region of the country.
# Display the plots across multiple rows.

# Create a regression plot with multiple rows
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           row="Region")

# Show the plot
plt.show()

# Create a regression plot with multiple columns
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           col="Region")

# Show the plot
plt.show()

