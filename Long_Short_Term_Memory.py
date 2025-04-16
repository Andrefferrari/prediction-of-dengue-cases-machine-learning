# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:30:20 2025

@author: andrefariasferrari
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
random.seed(137)
np.random.seed(137)
tf.random.set_seed(137)



#To read csv file we will import pandas library, import pandas as pd (already up there)
#Create a path so we can call the file
path = r"Set document path"
df = pd.read_csv(path, sep=";", decimal=",")
df.info()

#Vizualising outliers
df[["Data Medicao", "n_casos"]].plot(kind = "box")

#Locating outliers
df.n_casos.sort_values()

#Removing the last ten values from the list
top_outliers = list(df.n_casos.sort_values()[-10:].index)
top_outliers
df = df.drop(top_outliers)
len(df)

#Converting the date column into 'datetime" datatype
df['Data Medicao'] = pd.to_datetime(df['Data Medicao'])
df.info()

#Set date column as the index
df.set_index('Data Medicao', inplace = True)
df.info()
df.head()

#Set data into the ascending order, sequential order
#Sorting indexes
df.sort_index(inplace=True)

#Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_values = scaler.fit_transform(df[df.columns])

#Now lets put this data into a dataframe format
#Using pandas df, get the adjusted values and indicated the columns and the index
df_scaled = pd.DataFrame(scaler_values, columns = df.columns, index = df.index)
df_scaled

#Our next step is plotting the columns
#We can analyse the data patterns, etc.
plt.rcParams['figure.figsize'] = (15, 15) 
figure, axes = plt.subplots(9)

for ax, col in zip(axes, df_scaled.columns):
    ax.plot(df_scaled[col])
    ax.set_title(col)
    ax.axes.xaxis.set_visible(False)
    
#Creating the sliding window sequences
window_size = 6

def create_sequence(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)

X, y = create_sequence(df_scaled, window_size)
X.shape, y.shape


#Now we create the train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
X_train.shape, X_test.shape

#Building LSTM Model

model = keras.Sequential([
    #adding the first LSTM layer
    keras.layers.LSTM(units = 64, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])),
   
    #adding the second LSTM layer
    keras.layers.LSTM(units=32, return_sequences= True),
    keras.layers.Dropout(0.1),
    
    #adding the second LSTM layer
    keras.layers.LSTM(units=32, return_sequences= False),
    keras.layers.Dense(y_train.shape[1])
   ])

#Compilation
model.compile(optimizer='adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

#Train the model
lstm_model = model.fit(X_train, y_train,
                       validation_split = 0.2,
                       epochs = 100, 
                       batch_size = 5)  

lstm_model.history

#Predictions
predictions = model.predict(X_test)

#Then we need to rescale the data
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

#Analyse the metrics
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
r2_score(y_test_rescaled, predictions_rescaleded)
