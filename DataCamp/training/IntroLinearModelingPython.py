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

# Select a time not measured.
time = 8

# Use the model to compute a predicted distance for that time.
distance = model(time)

# Inspect the value of the predicted distance traveled.
print(distance)

# Determine if you will make it without refueling.
answer = (distance <= 400)
print(answer)

# Reasons for Modeling: Estimating Relationships
# Another common application of modeling is to compare two data sets by building models for each, and then comparing the models. In this exercise, you are given data for a road trip two cars took together. The cars stopped for gas every 50 miles, but each car did not need to fill up the same amount, because the cars do not have the same fuel efficiency (MPG). Complete the function efficiency_model(miles, gallons) to estimate efficiency as average miles traveled per gallons of fuel consumed. Use the provided dictionaries car1 and car2, which both have keys car['miles'] and car['gallons'].

# Complete the function definition for efficiency_model(miles, gallons).
# Use the function to compute the efficiency of the provided cars (dicts car1, car2).
# Store your answers as car1['mpg'] and car2['mpg'].
# Indicate which car has the best mpg by setting best_car=1, best_car=2, or best_car=0 if the same.

# Complete the function to model the efficiency.
def efficiency_model(miles, gallons):
   return np.mean(miles/gallons)

# Use the function to estimate the efficiency for each car.
car1['mpg'] = efficiency_model(car1['miles'], car1['gallons'])
car2['mpg'] = efficiency_model(car2['miles'], car2['gallons'])

# Finish the logic statement to compare the car efficiencies.
if car1['mpg'] > car2['mpg'] :
    print('car1 is the best')
elif car1['mpg'] < car2['mpg'] :
    print('car2 is the best')
else:
    print('the cars have the same efficiency')

# Object Interface

# Create figure and axis objects using subplots()
fig, axis = plt.subplots()

# Plot line using the axis.plot() method
line = axis.plot(times, distances , linestyle=" ", marker="o", color="red")

# Use the plt.show() method to display the figure
plt.show()

