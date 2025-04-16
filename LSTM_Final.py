import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
tf.random.set_seed(137)

#To read csv file we will import pandas library, import pandas as pd (already up there)

#Create a path so we can call the file
path = r"C:\Users\user\Documents\Estudos_Trabalhos_etc\MBA - USP_Esaql\TCC\Projeto_TCC\DTFinal1.xlsx1.csv"

df1 = pd.read_csv(path, sep=";", decimal=",")
df1.info()

train_dates = pd.to_datetime(df1['Data'])

#Converting the date column into 'datetime" datatype
df1['Data Medicao'] = pd.to_datetime(df1['Data'])
df1.info()

#Set date column as the index (inplace faz a mudança permanente)
df1.set_index('Data', inplace = True)
df1.info()

df1.head()

#Set data into the ascending order, sequential order
#Sorting indexes
df1.sort_index(inplace=True)

#Trying to Maks the Nans into 0
df1[np.isnan(df1)] = 0 


#data normalization
scaler = MinMaxScaler()
scaler_values1 = scaler.fit_transform(df1[df1.columns])

#Now lets put this data into a dataframe format
#Usando o pandas df, pegar os valores ajustados e indicar as colunas e index (mesmos do df original)
df1_scaled = pd.DataFrame(scaler_values1, columns = df1.columns, index = df1.index)

df1_scaled

#Our next step is plotting the columns
#We can analyse the data patterns etc
plt.rcParams['figure.figsize'] = (15, 15) 
figure, axes = plt.subplots(9) #numero de colunas do df

for ax, col in zip(axes, df_scaled.columns):
    ax.plot(df_scaled[col])
    ax.set_title(col)
    ax.axes.xaxis.set_visible(False)
    
#Creating the sliding window sequences
#window_size will be the number of time steps to predic the next one
window_size = 5

def create_sequence(data, window_size):
    W = []
    z = []
    for i in range(window_size, len(data)):
        W.append(data.iloc[i-window_size:i].values)
        z.append(data.iloc[i].values)
    return np.array(W), np.array(z)

W, z = create_sequence(df1_scaled, window_size)

W.shape, z.shape


#Now we create the train - test split
W_train, W_test, z_train, z_test = train_test_split(W, z, test_size=0.2, random_state=42, shuffle=False)
W_train.shape, W_test.shape

#Building LSTM Model

model = keras.Sequential([
    #adding the first LSTm layer
    keras.layers.LSTM(units = 50, return_sequences=True, input_shape = (W_train.shape[1], W_train.shape[2])),
    keras.layers.Dropout(0.1),
    
    #adding the second LSTm layer
    keras.layers.LSTM(units=25, return_sequences= False),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(z_train.shape[1])

   ])

#adding output layer
keras.layers.Dense(z_train.shape[1])

#Compilation
model.compile(optimizer='adam',
              loss = 'mean_squared_error')

#Early stopping
early_stopping = EarlyStopping(monitor = 'val_loss',
                               patience = 10,
                               restore_best_weights = True)

lstm_model = model.fit(W_train, z_train,
                       validation_split = 0.2,
                       epochs =80,
                       batch_size = 5,
                       callbacks = [early_stopping]) #hyper paramter (when reduced you get more precise output)

lstm_model.history


#Predictions
predictions1 = model.predict(W_test)

#Forecasting the data
#First we need to rescale the data
predictions_rescaleded1 = scaler.inverse_transform(predictions1)
z_test_rescaled = scaler.inverse_transform(z_test)

#predictions_reshaped = predictions.reshape(-1, 1) #transformar array 2d
#predictions_reshaped_nine = np.repeat(predictions_reshaped, df.shape[1], axis =-1)
#predictions_rescaled = scaler.inverse_transform(predictions_reshaped_nine)[:,0]
#y_test_rescaled = scaler.inverse_transform(y_test)


#Plotting the results
plt.figure(figsize = (20, 20))
for i, col in enumerate(df1_scaled.columns):
    plt.subplot(2, 14, i+1)
    plt.plot(z_test_rescaled[:,i], color = 'blue', label = f'Actual {col}')
    plt.plot(predictions_rescaleded1[:,i], color = 'red', label = f'Predicted{col}')
    plt.title(f'{col}Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()
    
    plt.tight_layout()





plt.figure(figsize = (12,8))
plt.plot(z_test_rescaled[:,0], label = "Valores reais", color='blue')
plt.plot(predictions_rescaleded1[:,0], label = "Predições", color='red')
plt.title("Teste")
plt.xlabel("Date")    
plt.ylabel("N_cases")

