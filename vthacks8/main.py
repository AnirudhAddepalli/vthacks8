from stockAnalysis.dataCollection import getData
from stockAnalysis.models import randomForestRegressor
from stockAnalysis import app
#from stockAnalysis.tdameritrade import get_movers






if __name__ == '__main__':
  

  #symbols = ['AAPL']
  #getData(symbols)

  #userStockInput = ['APPL']

  #stockInput('BTEC')

  app.run(host='0.0.0.0', port=8080)
  

  #randomForestRegressor('hist/AAPL.csv')