# Clustering 2D points
# From the scatter plot of the previous exercise, you saw that the points seem to separate into 3 clusters. You'll now create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise. After the model has been fit, you'll obtain the cluster labels for some new points using the .predict() method.
#
# You are given the array points from the previous exercise, and also an array new_points.
#
# Import KMeans from sklearn.cluster.
# Using KMeans(), create a KMeans instance called model to find 3 clusters. To specify the number of clusters, use the n_clusters keyword argument.
# Use the .fit() method of model to fit the model to the array of points points.
# Use the .predict() method of model to predict the cluster labels of new_points, assigning the result to labels.
# Hit 'Submit Answer' to see the cluster labels of new_points.

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

# Great work! You've successfully performed k-Means clustering and predicted the labels of new points. But it is not easy to inspect the clustering by just looking at the printed labels. A visualization would be far more useful. In the next exercise, you'll inspect your clustering with a scatter plot!

# Inspect your clustering
# Let's now inspect the clustering you performed in the previous exercise!
#
# A solution to the previous exercise has already run, so new_points is an array of points and labels is the array of their cluster labels.

# Import matplotlib.pyplot as plt.
# Assign column 0 of new_points to xs, and column 1 of new_points to ys.
# Make a scatter plot of xs and ys, specifying the c=labels keyword arguments to color the points by their cluster label. Also specify alpha=0.5.
# Compute the coordinates of the centroids using the .cluster_centers_ attribute of model.
# Assign column 0 of centroids to centroids_x, and column 1 of centroids to centroids_y.
# Make a scatter plot of centroids_x and centroids_y, using 'D' (a diamond) as a marker by specifying the marker parameter. Set the size of the markers to be 50 using s=50.

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c = labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

# Fantastic! The clustering looks great! But how can you be sure that 3 clusters is the correct choice? In other words, how can you evaluate the quality of a clustering? Tune into the next video in which Ben will explain how to evaluate a clustering!

# Inertia measures clustering quality
# Measures how spread out clusters are (lower is better)
# Distance from each sample to centroid of its cluster
# After fit available as attribute inertia_
# k-means attempts to minimise the inertia when choosing clusters
# scree plot is the inertia plot (inertia vs nr of clusters)
# Choose an elbow in the inertia plot where inertia begins to decrease more slowly

# How many clusters of grain?
# In the video, you learned how to choose a good number of clusters for a dataset using the k-means inertia graph. You are given an array samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. What's a good number of clusters in this case?
#
# KMeans and PyPlot (plt) have already been imported for you.
#
# This dataset was sourced from the UCI Machine Learning Repository.

# For each of the given values of k, perform the following steps:
# Create a KMeans instance called model with k clusters.
# Fit the model to the grain data samples.
# Append the value of the inertia_ attribute of model to the list inertias.
# The code to plot ks vs inertias has been written for you, so hit 'Submit Answer' to see the plot!

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Evaluating the grain clustering
# In the previous exercise, you observed from the inertia plot that 3 is a good number of clusters for the grain data. In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian". In this exercise, cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.
#
# You have the array samples of grain samples, and a list varieties giving the grain variety for each sample. Pandas (pd) and KMeans have already been imported for you.

# Create a KMeans model called model with 3 clusters.
# Use the .fit_predict() method of model to fit it to samples and derive the cluster labels. Using .fit_predict() is the same as using .fit() followed by .predict().
# Create a DataFrame df with two columns named 'labels' and 'varieties', using labels and varieties, respectively, for the column values. This has been done for you.
# Use the pd.crosstab() function on df['labels'] and df['varieties'] to count the number of times each grain variety coincides with each cluster label. Assign the result to ct.
# Hit 'Submit Answer' to see the cross-tabulation!

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

# Clustering not good? The wine features have very different variances
# In k-means clustering the variance of a features corresponds to its influence on the clustering algorithm
# Data needs to be transformed so that features have equal variance
# StandardScaler transforms each feature to have mean 0 and variance 1

# StandardScaler and KMeans have similar methods
# Use fit() / transform() with StandardScaler
# Use fit() / predict() with KMeans

# Use sklearn pipeline to combine multiple steps
# StandardScaler is a preprocessing step
# MaxAbsScaler and Normalizer are other examples

# Scaling fish data for clustering
# You are given an array samples giving measurements of fish. Each row represents an individual fish. The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. In order to cluster this data effectively, you'll need to standardize these features first. In this exercise, you'll build a pipeline to standardize and cluster the data.
#
# These fish measurement data were sourced from the Journal of Statistics Education.
#
# Import:
# make_pipeline from sklearn.pipeline.
# StandardScaler from sklearn.preprocessing.
# KMeans from sklearn.cluster.
# Create an instance of StandardScaler called scaler.
# Create an instance of KMeans with 4 clusters called kmeans.
# Create a pipeline called pipeline that chains scaler and kmeans. To do this, you just need to pass them in as arguments to make_pipeline().

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

# Great work! Now that you've built the pipeline, you'll use it in the next exercise to cluster the fish by their measurements.

# Clustering the fish data
# You'll now use your standardization and clustering pipeline from the previous exercise to cluster the fish by their measurements, and then create a cross-tabulation to compare the cluster labels with the fish species.
#
# As before, samples is the 2D array of fish measurements. Your pipeline is available as pipeline, and the species of every fish sample is given by the list species

# Import pandas as pd.
# Fit the pipeline to the fish measurements samples.
# Obtain the cluster labels for samples by using the .predict() method of pipeline.
# Using pd.DataFrame(), create a DataFrame df with two columns named 'labels' and 'species', using labels and species, respectively, for the column values.
# Using pd.crosstab(), create a cross-tabulation ct of df['labels'] and df['species'].




