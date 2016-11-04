
# coding: utf-8

# In[ ]:

#reading in data and importing all necessary packages
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import pandas as pd
bikes = pd.read_csv('../data/2016-Q1-Trips-History-Data.csv')
bikes.head()
bikes['start'] = pd.to_datetime(bikes['Start date'])
get_ipython().magic("time bikes['end'] = pd.to_datetime(bikes['End date'])")


# In[ ]:

bikes.head()


# In[ ]:

bikes['hour_of_day'] = (bikes.start.dt.hour + (bikes.start.dt.minute/60).round(2))

hours = bikes.groupby('hour_of_day').agg('count')
hours['hour'] = hours.index

hours.start.plot()

sns.lmplot(x='hour', y='start', data=hours, aspect=1.5, scatter_kws={'alpha':0.2})


# In[ ]:

hours[5:8].start.plot()

sns.lmplot(x='hour', y='start', data=hours[5:8], aspect=1.5, scatter_kws={'alpha':0.5})


# In[ ]:

# fit a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['al']
X = hours[['hour']]
y = hours.start
linreg.fit(X, y)


# In[ ]:

# putting the plots together
hours['pred'] = linreg.predict(X)

plt.scatter(hours.hour, hours.start)
plt.plot(hours.hour, hours.pred, color='red')
plt.xlabel('hours')
plt.ylabel('count')


# In[ ]:

# fitting a linear regression model
from sklearn.linear_model import LinearRegression
linreg = None
linreg = LinearRegression()

partial_hours = hours.loc[5.5:9]

X = partial_hours[['hour']]
y = partial_hours.start
linreg.fit(X, y)

hours.loc[5.5:9, 'pred'] = linreg.predict(partial_hours[['hour']])

# putting the plots together
plt.scatter(hours.hour, hours.start)
plt.plot(partial_hours.hour, partial_hours.pred, color='red')
plt.xlabel('hours')
plt.ylabel('count')


# In[ ]:



