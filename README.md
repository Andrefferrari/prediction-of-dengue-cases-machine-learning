# Prediction of dengue cases with machine and deep learning

## Overview
Dengue is a serious health issue that affects the majority of tropical countries and should be treated serioulsy. Climate changes won't stop happening and the world is heating more with the passing of each year, sea levels are rising and many and other natural accidents keep increasing, wich for these countrys means more ocurrences. And the area for the insect distribution keeps increasing.

This project obtained oficial data of confirmed dengue cases for the municipality of São Paulo (Brazil) through the Unified Health System (SUS) and climate data from the National Institute of Meteorology (INMET), with nine climate variables. These two datas were combined together on a single dataset by monthly cases and climate observations for predictions.

## Objectives
Creating Machine Learning and Deep Learning models for data prediction of number of dengue cases on the municipality of São Paulo through climate data using the last six months and alaysing the best model.

## Methodology

- **Preprocessing the data**: Sampling the data and scaling when required by the model, removing outliers 
- **Importance check**: Using Grid methods for sampling the best variables for use  
- **Hyperparameter Optimization**: Fine-tunning the model for better prediction  
- **Results Analysis**: Comparing predictions with origial test data, and choosing the model with more accurate data  

## Accuracy Assessment
The following metrics were used for prediction analysis:
- **Mean Absolute Error (MAE)**  
- **Root Mean Square Error (RMSE)**  
- **Coefficient of Determination (R²)**  

## Results  
- **Random Forest**:  
MAE: 267  
RMSE: 487  
R²: 21%%  
- **Long Short-Term Memory**:   
MAE: 108  
RMSE: 493  
R²: 54%  

The best model was Long Short-Term Memory with a more accurate MAE and best Coefficient of Determination, a sattisfatory result for the target feature. The dataset has proven to be smaller than desirable and therefore had difficulties of overfitting and generalization, more variables as a bigger sample could improve the models predictions.
