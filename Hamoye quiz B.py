#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
df= pd.read_csv(r'C:/Users/Victor/Downloads/energy.csv')
df


# In[17]:


df.isnull().sum()


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[3]:


X = df[['T2']].values 
y = df[['T6']].values


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


y_pred = model.predict(X_test)


# In[7]:


rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE): {:.3f}".format(rmse))


# In[23]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# In[9]:


df = df.drop(columns=["date", "lights"])


# In[11]:


X = df.drop(columns=["Appliances"]).values
y = df["Appliances"].values


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[18]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[19]:


model = LinearRegression()
model.fit(X_train_scaled, y_train)


# In[26]:


y_train_pred = model.predict(X_train_scaled)


# In[27]:


mae_train = mean_absolute_error(y_train, y_train_pred)
print("Mean Absolute Error (MAE) for the training set: {:.3f}".format(mae_train))


# In[29]:


y_train_pred = model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5
print("Root Mean Squared Error (RMSE) for the training set: {:.3f}".format(rmse_train))


# In[ ]:




