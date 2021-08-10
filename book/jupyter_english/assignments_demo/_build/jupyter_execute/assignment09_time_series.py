#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# 
# Author: Mariya Mansurova, Analyst & developer in Yandex.Metrics team. Translated by Ivan Zakharov, ML enthusiast. <br>This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center> Assignment #9 (demo)
# ## <center> Time series analysis
# 
# **Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a9-demo-time-series-analysis) + [solution](https://www.kaggle.com/kashnitsky/a9-demo-time-series-analysis-solution).**
# 
# **Fill cells marked with "Your code here" and submit your answers to the questions through the [web form](https://docs.google.com/forms/d/1UYQ_WYSpsV3VSlZAzhSN_YXmyjV7YlTP8EYMg8M8SoM/edit).**

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import requests

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


# **<font color='red'>Question 1:</font>** What is the prediction of the number of views of the wiki page on January 20? Round to the nearest integer.
# 
# - 4947
# - 3426
# - 5229
# - 2744

# In[8]:


# You code here


# Estimate the quality of the prediction with the last 30 points.

# In[9]:


# You code here


# **<font color='red'>Question 2:</font> What is MAPE equal to?**
# 
# - 34.5
# - 42.42
# - 5.39
# - 65.91
# 
# **<font color='red'>Question 3:</font> What is MAE equal to?**
# 
# - 355
# - 4007
# - 600
# - 903

# ## Predicting with ARIMA

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (15, 10)


# **<font color='red'>Question 4:</font> Let's verify the stationarity of the series using the Dickey-Fuller test. Is the series stationary? What is the p-value?**
# 
# - Series is stationary, p_value = 0.107
# - Series is not stationary, p_value = 0.107
# - Series is stationary, p_value = 0.001
# - Series is not stationary, p_value = 0.001

# In[11]:


# You code here


# **Next, we turn to the construction of the SARIMAX model (`sm.tsa.statespace.SARIMAX`).<br> <font color='red'>Question 5:</font> What parameters are the best for the model according to the `AIC` criterion?**
# 
# - D = 1, d = 0, Q = 0, q = 2, P = 3, p = 1
# - D = 2, d = 1, Q = 1, q = 2, P = 3, p = 1
# - D = 1, d = 1, Q = 1, q = 2, P = 3, p = 1
# - D = 0, d = 0, Q = 0, q = 2, P = 3, p = 1

# In[12]:


# You code here

