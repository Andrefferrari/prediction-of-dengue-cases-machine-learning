# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:38:07 2025

@author: andrefariasferrari
"""
#Install all the needed libraries/packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from matplotlib.pyplot import figure

#Create a path so we can call the file
path = r"Set document path"
df = pd.read_csv(path, sep=";", decimal=",")
df.info()

#Analysing outliers
df[["Data Medicao", "n_casos"]].plot(kind = "box")

#Locating outliers
df.n_casos.sort_values()

#Removing the las forty values from the list
top_outliers = list(df.n_casos.sort_values()[-40:].index)
top_outliers
df = df.drop(top_outliers)
len(df)

#Separating the target from the other variables
y = df.loc[:,'n_casos']
X = df.drop(['n_casos'], axis='columns')

#Splitting the data between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Training the model for the first time
rfr = RandomForestRegressor(n_jobs= -1, random_state=42)

#First we analyse the importances
rfModel= rfr.fit(X_train, y_train)

#Importance
importance = rfModel.feature_importances_
importance

#Plotting the importance for analysis
X_train.columns
columns = X_train.columns
rFGraph = pd.Series(importance, columns)
rFGraph

#Now we start analysing the model without each of the variables
#Dropping the date
df = df.drop(['Data Medicao'], axis='columns')

#Variables listed from higher to lower importance level
df = df.drop(['UMIDADE RELATIVA DO AR, MEDIA MENSAL(%)'], axis='columns')
df = df.drop(['PRECIPITACAO TOTAL, MENSAL(mm)'], axis='columns')
df = df.drop(['TEMPERATURA MINIMA MEDIA, MENSAL(Â°C)'], axis='columns')
df = df.drop(['EVAPOTRANSPIRACAO POTENCIAL, BH MENSAL(mm)'], axis='columns')
df = df.drop(['EVAPOTRANSPIRACAO REAL, BH MENSAL(mm)'], axis='columns')
df = df.drop(['VENTO, VELOCIDADE MEDIA MENSAL(m/s)'], axis='columns')
df = df.drop(['TEMPERATURA MEDIA COMPENSADA, MENSAL(Â°C)'], axis='columns')
df = df.drop(['TEMPERATURA MAXIMA MEDIA, MENSAL(Â°C)'], axis='columns')


#Then back to the process until we find the best combination
#Create a path so we can call the file
path = r"Set document path"  
df = pd.read_csv(path, sep=";", decimal=",")
df.info()

#Vizualising outliers
df[["Data Medicao", "n_casos"]].plot(kind = "box")

#Locating outliers
df.n_casos.sort_values()

#Removing the last forty values from the list
top_outliers = list(df.n_casos.sort_values()[-40:].index)
top_outliers
df = df.drop(top_outliers)
len(df)

#Separating the target from the other variables
y = df.loc[:,'n_casos']
X = df.drop(['n_casos'], axis='columns')

#Now we create the train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Training the model
rfr = RandomForestRegressor(n_jobs= -1, random_state=42)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_test)

#Analyse the metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
r2_score(y_test, y_pred)

#Tunning the model / analysing best paramethers
Param_total = {'n_estimators':[20, 30, 50],
               'max_depth': [1, 10, 15, 20],
               'max_features': [1, 5, 10],
               'min_samples_split': [10,  30,  50],
               'min_samples_leaf': [ 20, 30, 40, 50],
               'max_leaf_nodes': [2, 4, 8]
               }

rfr_cv = GridSearchCV(estimator=rfr, param_grid=Param_total, n_jobs=-1)
rfr_cv.fit(X_train, y_train)

rfr_cv.best_params_

y_pred = rfr_cv.predict(X_test)

#Analyse the "tunned" metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
r2_score(y_test, y_pred)

#Test the best results
rfr = RandomForestRegressor(n_estimators = 50, max_depth = 15, max_features = 5, max_leaf_nodes = 8, min_samples_split = 10, min_samples_leaf = 20, n_jobs= -1, random_state=42)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
