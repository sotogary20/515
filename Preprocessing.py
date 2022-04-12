# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import pandas as pd

file = './Data/fer2013.csv'
Width = 48
Height = 48

data = pd.read_csv(file)

data_public_test = data[data.Usage == 'PublicTest']
data_priv_test = data[data.Usage == 'PrivateTest']
data = data[data.Usage == 'Training']

Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def fer2013_to_X(data):
    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array
    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""

    X = []
    pixels_list = data["pixels"].values

    for pixels in pixels_list:
        single_image = np.reshape(pixels.split(" "), (Width, Height)).astype("float")
        X.append(single_image)

    # Convert list to 4D array:
    X = np.expand_dims(np.array(X), -1)

    # Normalize image data:
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    return X

X = fer2013_to_X(data)
X_public = fer2013_to_X(data_public_test)
X_private = fer2013_to_X(data_priv_test)
y = pd.get_dummies(data['emotion'].values)
y_pub = pd.get_dummies(data_public_test['emotion'].values)
y_priv = pd.get_dummies(data_priv_test['emotion'].values)
np.save('./Data/fer2013_X', X)
np.save('./Data/fer2013_X_Pub', X_public)
np.save('./Data/fer2013_X_Priv', X_private)
np.save('./Data/fer2013_y', y)
np.save('./Data/fer2013_y_pub', y_pub)
np.save('./Data/fer2013_y_priv', y_priv)