# Exploring the NSFG data
# To get the number of rows and columns in a DataFrame, you can read its shape attribute.
#
# To get the column names, you can read the columns attribute. The result is an Index, which is a Pandas data structure that is similar to a list. Let's begin exploring the NSFG data! It has been pre-loaded for you into a DataFrame called nsfg.

# value_counts() to see what values appear in pounds and how many times each value appears. By default results are sorted with most freq value first so using sort_index() sorts them by value instead

# describe() computes summary stats

# replace takes a list of values we want to replace and the value we want to replace them with

pounds = pounds.replace([98, 99], np.nan) # np.nan means we are getting the special value NaN from the NumPy library

# Clean a variable
# In the NSFG dataset, the variable 'nbrnaliv' records the number of babies born alive at the end of a pregnancy.
#
# If you use .value_counts() to view the responses, you'll see that the value 8 appears once, and if you consult the codebook, you'll see that this value indicates that the respondent refused to answer the question.
#
# Your job in this exercise is to replace this value with np.nan. Recall from the video how Allen replaced the values 98 and 99 in the ounces column using the .replace() method:
#
# ounces.replace([98, 99], np.nan, inplace=True)

# In the 'nbrnaliv' column, replace the value 8, in place, with the special value NaN.
# Confirm that the value 8 no longer appears in this column by printing the values and their frequencies.

# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace([8], np.nan, inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())

# Compute a variable
# For each pregnancy in the NSFG dataset, the variable 'agecon' encodes the respondent's age at conception, and 'agepreg' the respondent's age at the end of the pregnancy.
#
# Both variables are recorded as integers with two implicit decimal places, so the value 2575 means that the respondent's age was 25.75.

# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())

# Good job. A variable that's computed from other variables is sometimes called a 'recode'. It's now time to get back to the motivating question for this chapter: what is the average birth weight for babies in the U.S.? See you in the next video!

# ~ operator is logical NOT or inverse, for instance: full_term_weight = birth_weight[~preterm]

# Make a histogram
# Histograms are one of the most useful tools in exploratory data analysis. They quickly give you an overview of the distribution of a variable, that is, what values the variable can have, and how many times each value appears.
#
# As we saw in a previous exercise, the NSFG dataset includes a variable 'agecon' that records age at conception for each pregnancy. Here, you're going to plot a histogram of this variable. You'll use the bins parameter that you saw in the video, and also a new parameter - histtype - which you can read more about here in the matplotlib documentation. Learning how to read documentation is an essential skill. If you want to learn more about matplotlib, you can check out DataCamp's Introduction to Matplotlib course.

# Plot the histogram
plt.hist(agecon, bins=20, histtype = 'step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

# Compute birth weight
# Now let's pull together the steps in this chapter to compute the average birth weight for full-term babies.
#
# I've provided a function, resample_rows_weighted, that takes the NSFG data and resamples using the sampling weights in wgt2013_2015. The result is a DataFrame, sample, that is representative of the U.S. population.
#
# Then I extract birthwgt_lb1 and birthwgt_oz1, replacing special codes with NaN:
#
# # Resample the data
# sample = resample_rows_weighted(nsfg, 'wgt2013_2015')
#
# # Clean the weight variables
# pounds = sample['birthwgt_lb1'].replace([98, 99], np.nan)
# ounces = sample['birthwgt_oz1'].replace([98, 99], np.nan)

# Use pounds and ounces to compute total birth weight.
# Make a Boolean Series called preterm that is true for babies with 'prglngth' less than 37 weeks.
# Use preterm to select birth weight for babies that are not preterm. Store the result in full_term_weight.
# Compute the mean weight of full term babies.

# Compute total birth weight
birth_weight = pounds + ounces/16

# Create a Boolean Series for preterm babies
preterm = sample['prglngth'] < 37

# Select the weights of full term babies
full_term_weight = birth_weight[~preterm]

# Compute the mean weight of full term babies
print(np.mean(full_term_weight))

# Filter
# In the previous exercise, you computed the mean birth weight for full-term babies; you filtered out preterm babies because their distribution of weight is different.
#
# The distribution of weight is also different for multiple births, like twins and triplets. In this exercise, you'll filter them out, too, and see what effect it has on the mean.

# Use the variable 'nbrnaliv' to make a Boolean Series that is True for single births (where 'nbrnaliv' equals 1) and False otherwise.
# Use Boolean Series and logical operators to select single, full-term babies and compute their mean birth weight.
# For comparison, select multiple, full-term babies and compute their mean birth weight.

