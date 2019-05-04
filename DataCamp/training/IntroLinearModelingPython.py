# Reasons for Modeling: Interpolation
# One common use of modeling is interpolation to determine a value "inside" or "in between" the measured data points. In this exercise, you will make a prediction for the value of the dependent variable distances for a given independent variable times that falls "in between" two measurements from a road trip, where the distances are those traveled for the given elapse times.
#
# context figure
#
# Instructions
# 100 XP
# Inspect the predefined data arrays, times and distances, and the preloaded plot.
# Based on your rough inspection, estimate the distance_traveled away from the starting position as of elapse_time = 2.5 hours.
# Assign your answer to distance_traveled.

# Compute the total change in distance and change in time
total_distance = distances[-1] - distances[0]
total_time = times[-1] - times[0]

# Estimate the slope of the data from the ratio of the changes
average_speed = total_distance / total_time

# Predict the distance traveled for a time not measured
elapse_time = 2.5
distance_traveled = average_speed * elapse_time
print("The distance traveled is {}".format(distance_traveled))

# Correct! Notice that the answer distance is 'inside' that range of data values, so, less than the max(distances) but greater than the min(distances)

# Reasons for Modeling: Extrapolation
# Another common use of modeling extrapolation to estimate data values "outside" or "beyond" the range (min and max values of time) of the measured data. In this exercise, we have measured distances for times 0 through 5 hours, but we are interested in estimating how far we'd go in 8 hours. Using the same data set from the previous exercise, we have prepared a linear model distance = model(time). Use that model() to make a prediction about the distance traveled for a time much larger than the other times in the measurements.

# Use distance = model(time) to extrapolate beyond the measured data to time=8 hours.
# Print the distance predicted and then check whether it is less than or equal to 400.
# If your car can travel, at most, 400 miles on a full tank, and it takes 8 hours to drive home, will you make it without refilling? You should have answer=True if you'll make it, or answer=False if you will run out of gas.

