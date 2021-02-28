from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from stockAnalysis.dataCollection import getData
from PIL import Image



def randomForestRegressor(dataset):
 #Importing data and creating datasets

  data = pd.read_csv(dataset)
  data = data[['Adj Close']]
  #print(data)
  #print(data.head())

  forecast = 30
  data['Prediction'] = data[['Adj Close']].shift(-forecast)
  
  X = np.array(data.drop(['Prediction'], 1))
  X = preprocessing.scale(X)

  X_forecast = X[-forecast:]
  X = X[:-forecast]

  y = np.array(data['Prediction'])
  y = y[:-forecast]


  #Splitting Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  #Creating Model
  regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

  #Training Model
  regressor.fit(X, y)

  #Evaluating
  confidence = regressor.score(X_test, y_test)
  #print(confidence)

  forecast_predicted= regressor.predict(X_forecast)
  '''
  fig, (ax1, ax2) = plt.subplots(1,2)
  fig.suptitle('Historical Price Chart (left) and \n Price Prediction (right)')
  ax1.plot(data['Adj Close'])
  ax2.plot(forecast_predicted)'''
  #plt.show()

  return forecast_predicted

def stockInput(stocks):
  getData([stocks])
  arrayPredictions = np.around(randomForestRegressor('hist/{}.csv'.format(stocks)), 2)
  listPredictions = arrayPredictions.tolist()
  converted_list = [str(element) for element in listPredictions]
  count =0
  day=1
  for values in converted_list:
    converted_list[count]= " Day {}:".format(day) + " $"+values
    count += 1
    day += 1
  stringPredictions = ", \n".join(converted_list)
  return stringPredictions


'''def stockInput(stocks):
  getData([stocks])
  arrayPredictions = np.around(randomForestRegressor('hist/{}.csv'.format(stocks)), 2)
  stringPredictions = np.array_str(arrayPredictions)
  return stringPredictions
  '''