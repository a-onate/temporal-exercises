#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os
import seaborn as sns


# In[3]:


path = "data/ex1data1.txt"


# In[8]:


data = pd.read_csv(path, names =['Population', 'Profit'] )


# In[10]:


from sklearn import linear_model
model = linear_model.LinearRegression()
X=np.array(data['Population']).reshape((-1,1))
y=np.array(data['Profit'])
model.fit(X, y)


# In[18]:


x = X[:, 0]


# In[19]:


f = model.predict(X).flatten()


# In[20]:


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')


# In[ ]:




