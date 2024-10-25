import pandas as pd

data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
data = data.drop('Fremont Bridge Sidewalks, south of N 34th St',axis=1)
data.head(n=10)
data.plot()