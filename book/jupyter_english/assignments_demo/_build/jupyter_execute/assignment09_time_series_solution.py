#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# 
# Author: Mariya Mansurova, Analyst & developer in Yandex.Metrics team. Translated by Ivan Zakharov, ML enthusiast. <br>This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center> Assignment #9 (demo). Solution
# ## <center> Time series analysis
# 
# **Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a9-demo-time-series-analysis) + [solution](https://www.kaggle.com/kashnitsky/a9-demo-time-series-analysis-solution).**
# 
# **Fill cells marked with "Your code here" and submit your answers to the questions through the [web form](https://docs.google.com/forms/d/1UYQ_WYSpsV3VSlZAzhSN_YXmyjV7YlTP8EYMg8M8SoM/edit).**

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import requests
import pandas as pd

print(__version__) # need 1.9.0 or greater
init_notebook_mode(connected = True)


def plotly_df(df, title = ''):
    data = []
    
    for column in df.columns:
        trace = go.Scatter(
            x = df.index,
            y = df[column],
            mode = 'lines',
            name = column
        )
        data.append(trace)
    
    layout = dict(title = title)
    fig = dict(data = data, layout = layout)
    iplot(fig, show_link=False)


# ## Data preparation

# In[2]:


df = pd.read_csv('../../data/wiki_machine_learning.csv', sep = ' ')
df = df[df['count'] != 0]
df.head()


# In[3]:


df.shape


# ## Predicting with FB Prophet
# We will train at first 5 months and predict the number of trips for June.

# In[4]:


df.date = pd.to_datetime(df.date)


# In[5]:


plotly_df(df.set_index('date')[['count']])


# In[6]:


from fbprophet import Prophet


# In[7]:


predictions = 30

df = df[['date', 'count']]
df.columns = ['ds', 'y']
df.tail()


# In[8]:


train_df = df[:-predictions].copy()


# In[9]:


m = Prophet()
m.fit(train_df);


# In[10]:


future = m.make_future_dataframe(periods=predictions)
future.tail()


# In[11]:


forecast = m.predict(future)
forecast.tail()


# **<font color='red'>Question 1:</font>** What is the prediction of the number of views of the wiki page on January 20? Round to the nearest integer.
# 
# - 4947
# - 3426 **[+]**
# - 5229
# - 2744

# In[12]:


m.plot(forecast)


# In[13]:


m.plot_components(forecast)


# In[14]:


cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))


# In[15]:


cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
cmp_df['p'] = 100 * cmp_df['e'] / cmp_df['y']
print('MAPE = ', round(np.mean(abs(cmp_df[-predictions:]['p'])), 2))
print('MAE = ', round(np.mean(abs(cmp_df[-predictions:]['e'])), 2))


# Estimate the quality of the prediction with the last 30 points.
# 
# **<font color='red'>Question 2:</font> What is MAPE equal to?**
# 
# - 34.5 **[+]**
# - 42.42
# - 5.39
# - 65.91
# 
# **<font color='red'>Question 3:</font> What is MAE equal to?**
# 
# - 355
# - 4007
# - 600 **[+]**
# - 903

# ## Predicting with ARIMA

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (15, 10)


# **<font color='red'>Question 4:</font> Let's verify the stationarity of the series using the Dickey-Fuller test. Is the series stationary? What is the p-value?**
# 
# - Series is stationary, p_value = 0.107
# - Series is not stationary, p_value = 0.107 **[+]**
# - Series is stationary, p_value = 0.001
# - Series is not stationary, p_value = 0.001

# In[17]:


sm.tsa.seasonal_decompose(train_df['y'].values, freq=7).plot();
print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(train_df['y'])[1])


# But the seasonally differentiated series will already be stationary.

# In[18]:


train_df.set_index('ds', inplace=True)


# In[19]:


train_df['y_diff'] = train_df.y - train_df.y.shift(7)
sm.tsa.seasonal_decompose(train_df.y_diff[7:].values, freq=7).plot();
print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(train_df.y_diff[8:])[1])


# In[20]:


ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(train_df.y_diff[13:].values.squeeze(), lags=48, ax=ax)

ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(train_df.y_diff[13:].values.squeeze(), lags=48, ax=ax)


# Initial values:
# * Q = 1
# * q = 3
# * P = 3
# * p = 1

# In[21]:


ps = range(0, 2)
ds = range(0, 2)
qs = range(0, 4)
Ps = range(0, 4)
Ds = range(0, 3)
Qs = range(0, 2)


# In[22]:


from itertools import product

parameters = product(ps, ds, qs, Ps, Ds, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[23]:


get_ipython().run_cell_magic('time', '', 'import warnings\nfrom tqdm import tqdm\nresults1 = []\nbest_aic = float("inf")\nwarnings.filterwarnings(\'ignore\')\n\nfor param in tqdm(parameters_list):\n    #try except is necessary, because on some sets of parameters the model can not be trained\n    try:\n        model=sm.tsa.statespace.SARIMAX(train_df[\'y\'], order=(param[0], param[1], param[2]), \n                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)\n    #print parameters on which the model is not trained and proceed to the next set\n    except (ValueError, np.linalg.LinAlgError):\n        continue\n    aic = model.aic\n    #save the best model, aic, parameters\n    if aic < best_aic:\n        best_model = model\n        best_aic = aic\n        best_param = param\n    results1.append([param, model.aic])')


# In[24]:


result_table1 = pd.DataFrame(results1)
result_table1.columns = ['parameters', 'aic']
print(result_table1.sort_values(by = 'aic', ascending=True).head())


# If we consider the variants proposed in the form:

# In[25]:


result_table1[result_table1['parameters'].isin([(1, 0, 2, 3, 1, 0),
                                                (1, 1, 2, 3, 2, 1),
                                                (1, 1, 2, 3, 1, 1),
                                                (1, 0, 2, 3, 0, 0)])]


# Now do the same, but for the series with Box-Cox transformation.

# In[26]:


import scipy.stats
train_df['y_box'], lmbda = scipy.stats.boxcox(train_df['y']) 
print("The optimal Box-Cox transformation parameter: %f" % lmbda)


# In[27]:


results2 = []
best_aic = float("inf")

for param in tqdm(parameters_list):
    #try except is necessary, because on some sets of parameters the model can not be trained
    try:
        model=sm.tsa.statespace.SARIMAX(train_df['y_box'], order=(param[0], param[1], param[2]), 
                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)
    #print parameters on which the model is not trained and proceed to the next set
    except (ValueError, np.linalg.LinAlgError):
        continue
    aic = model.aic
    #save the best model, aic, parameters
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results2.append([param, model.aic])
    
warnings.filterwarnings('default')


# In[28]:


result_table2 = pd.DataFrame(results2)
result_table2.columns = ['parameters', 'aic']
print(result_table2.sort_values(by = 'aic', ascending=True).head())


# If we consider the variants proposed in the form:

# In[29]:


result_table2[result_table2['parameters'].isin([(1, 0, 2, 3, 1, 0),
                                                (1, 1, 2, 3, 2, 1),
                                                (1, 1, 2, 3, 1, 1),
                                                (1, 0, 2, 3, 0, 0)])].sort_values(by='aic')


# **Next, we turn to the construction of the SARIMAX model (`sm.tsa.statespace.SARIMAX`).<br> <font color='red'>Question 5:</font> What parameters are the best for the model according to the `AIC` criterion?**
# 
# - D = 1, d = 0, Q = 0, q = 2, P = 3, p = 1
# - D = 2, d = 1, Q = 1, q = 2, P = 3, p = 1 **[+]**
# - D = 1, d = 1, Q = 1, q = 2, P = 3, p = 1
# - D = 0, d = 0, Q = 0, q = 2, P = 3, p = 1

# Let's look at the forecast of the best AIC model.

# In[30]:


print(best_model.summary())


# In[31]:


plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Student's test: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])


# In[32]:


def invboxcox(y,lmbda):
    # reverse Box Cox transformation
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda * y + 1) / lmbda))


# In[33]:


train_df['arima_model'] = invboxcox(best_model.fittedvalues, lmbda)

train_df.y.tail(200).plot()
train_df.arima_model[13:].tail(200).plot(color='r')
plt.ylabel('wiki pageviews');

