### ****Epoch Core Task-1****
***PS*** :\
You have been provided with a dataset containing information about various pincodes across India, including their corresponding longitudes and latitudes (clustering_data.csv). Your task is to focus specifically on the pincodes of your home state.\
***Procedure*** :\
Libraries used : numpy, pandas, matplotlib, contextily, Geopandas\
**DATA PREPROCESSING** :\
1. Setting NA values to NaN
2. Stripping values to remove extra spaces
3. Converting using to_numeric
4. Using .interpolate() to fill NaN values - it estimates missing values using surrounding data.\
**DATA VISUALISATION** :\
1. Creating a GeoPandas dataframe in which geometry is set to longitude and latitude resp.
2. Setting Coordinate Reference System(CRS) to WGS 84(World Geodetic System 1984) using EPSG:4326. WGS 84 uses latitude and longitude in degrees to represent locations on Earth.
3. We add the basemap by taking the source from OpenStreetMap.Mapnik and set the limits to focus on Andhra Pradesh.\
**K-MEANS CLUSTERING** :\
Steps in K-Means :\
1. Assign random centroids from a given range.
2. Calculate distances from centroids.
3. Assign label using 'argmin' of distances.
4. Updata centroids by calculating mean of points.
5. Iterate this till centroids reach a point of convergence.\
**VISUZLISATION AND INFERENCES** :\
1. We create a dataframe for centroids in the similar way as before.
2. We iterate through label numbers and collect data of same cluster, then plot this data.
3. Since it is the postal pincodes we are plotting, it helps us to recognise the density of living and also gives an idea if the area is underdeveloped.\
***That's it! This ends our model!***
### ***Task-2***
***PS*** :\
The primary objective of this project is to use artificial intelligence to convert handwritten text images into digital text and subsequently perform sentiment analysis on the extracted text.\
***Procedure*** :\
Libraries used : numpy, pandas, matplotlib, tensorflow, opencv, math, scikit-learn\
***OCR MODEL*** :\
I preferred CNN here and for that we need to import certain modules - from tensorflow.keras.models import Sequential\
                                                                     \from sklearn.preprocessing import LabelEncoder\
                                                                     \from sklearn.model_selection import train_test_split\
                                                                     \from tensorflow.keras.utils import to_categorical\
                                                                     \from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout\
 **PREPROCESSING** :\
 We filter the data by considering only rows containing label as uppercase-alphabet. Then we assign labels as first column and the rest is converted to numpy array adn reshaped as (28,28,1) which means, 28*28 pixel data with 1 showing greyscale.\
 The values are then *#normalized and stored as 'images'.\
 **MODEL** :\
 This has many layers.\
 Conv2D : This is like building block of CNN and performs convolution operation. Here the first argument is number of filters and next (3,3) is kernel size. This creates the convolutional matrix and the input_shape we provided is (28,28,1). The activation funciton is set to 'relu' which is Rectified Linear Unit, which sets to 1 if positive, 0 otherwise. Setting padding to same means that output from this layer has same dimentions as input layer. This is achieved by padding zeros to input layer.\
Flatten : This layer flattens the output of the convolutional layer into a 1D feature vector. This is necessary because the output of the convolutional layer is a 3D tensor (height, width, channels), but the next layer (dense layer) expects a 1D input.\
MaxPooling2D :  A max pooling layer, which downsamples the input data to reduce spatial dimensions and the number of parameters.\
DropOut :  A dropout layer, which randomly sets a fraction of the neurons to zero during training to prevent overfitting.\
Dense : This is a fully connected layer. Here the first dense layer contains 64 neurons, the role of this is to ensure that the feature representation is in the correct format since the output layer expects a 1D feature vector as input. The next dense layer is the output layer having 26 neurons, the activation function used here is 'softmax', which is used to output a probability distribution over the classes.\
**Next** is to compile, we use 'adam' optimiser here to update the parameters of model during training. Since our data here can be classified as categorical, loss is set to 'categorical_crossentropy', for which the corresponding metric would be 'accuracy'. In case data is sparse, we consider loss as 'sparse_categorical_crossentropy'.
*Callback* : We create a TensorBoard callback object in Keras, which is used to visualize and track the training process of a deep learning model. In this part, log_dir is the directory in which the log info is stored. This includes Loss and accuracy values for each epoch, Learning rate schedules, Model weights and gradients, etc. After the process is complete, it can be visualised by running 'tensorboard --logdir=./logs' on terminal.\
*Label processing* : Since the labels we have are alphabets, we might need to collect corresponding numbers since our nn returns label as number. For that we create a dictionary 'label_mapping' and collect 'numerical_labels' from that.Then these are 'one-hot coded' using to_categorical function, which basically means we generate a 28*28 array showing each label. This is especially useful here when we are using soft-max algorithm.\
*Then our model is split, trained and evaluated*\
***SENTIMENT ANALYSIS***
Here we create a naive-bayes model for sentiment analysis. First I have done it using MultinomialNB directly, then defined own function for each.\
Extra libraries imported- nltk : For preprocessing data, removing stop words and punctuations. Ofcourse, we can do this even without the library.\
**PREPROCESSING** :\
Data is extracted and sentiments are mapped as happy -1, angry- -1 and neutral-0.\
In the process using MultinomialNB, we use CountVectorizer to generate sparse matrix containing the vectorized form of our data.\
We created two functions here- clean_text to process our data and feature to extract distinct words from that data.
***MODEL*** :
Here's how *naive bayes* works,
1. Important thing here is word count, it basically works on just that parameter, which is the reason why it is very less accurate.
2. Just Bayes formula is used, P(class/features) = P(features/class)*P(class) / P(feature), P(features) is the constant, so there's no need for it while comparing probabilities. So, P(class/features) = P(features/class)*P(class). Here feature used is word count.\
3. P(features/class) again goes down to product of individual probabilities.\
4. In our first function, 'train_naive_bayes' we calculate the P(features/class) and in the next it is used to find P(class/features). Accuracy, precision, recall and F1-score are then calculated for y_pred and y_test.\
5. As the metrics show, accuracy is very less (as expected...)\
***With this we completed our sentiment analysis model!***\
Next is to test these combined on our target_images!\
For this,we use opencv library, first we read the image using imread, resize it, then convert into a numpy array, then reshape it as we want! (1,28,28,1) to match it though..\
Then appy our funcitons for naive bayes and find the sentiment.\
