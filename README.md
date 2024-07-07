#### ***Epoch Core Task-1***
****PS**** : \
You have been provided with a dataset containing information about various pincodes across India, including their corresponding longitudes and latitudes (clustering_data.csv). Your task is to focus specifically on the pincodes of your home state.\\ 
****Procedure**** : \
Libraries used : numpy, pandas, matplotlib, contextily, Geopandas \
***DATA PREPROCESSING*** : \
1. Setting NA values to NaN
2. Stripping values to remove extra spaces
3. Converting using to_numeric
4. Using .interpolate() to fill NaN values - it estimates missing values using surrounding data. \
***DATA VISUALISATION*** : \
1. Creating a GeoPandas dataframe in which geometry is set to longitude and latitude resp.
2. Setting Coordinate Reference System(CRS) to WGS 84(World Geodetic System 1984) using EPSG:4326. WGS 84 uses latitude and longitude in degrees to represent locations on Earth.
3. We add the basemap by taking the source from OpenStreetMap.Mapnik and set the limits to focus on Andhra Pradesh.
***K-MEANS CLUSTERING*** : \
Steps in K-Means : \
1. Assign random centroids from a given range.
2. Calculate distances from centroids.
3. Assign label using 'argmin' of distances.
4. Updata centroids by calculating mean of points.
5. Iterate this till centroids reach a point of convergence.
***VISUZLISATION AND INFERENCES*** : \
1. We create a dataframe for centroids in the similar way as before.
2. We iterate through label numbers and collect data of same cluster, then plot this data.
3. Since it is the postal pincodes we are plotting, it helps us to recognise the density of living and also gives an idea if the area is underdeveloped.

#### ***Task-2***
****PS**** : \
The primary objective of this project is to use artificial intelligence to convert handwritten text images into digital text and subsequently perform sentiment analysis on the extracted text. \
****Procedure**** : \
Libraries used : numpy, pandas, matplotlib, tensorflow, opencv, re, math, scikit-learn \
***DATA PREPROCESSING*** : \ 
