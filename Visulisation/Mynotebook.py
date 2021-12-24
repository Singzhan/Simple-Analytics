#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python --version')


# In[2]:


get_ipython().system('jupyter --version')


# In[4]:


get_ipython().system('pip install seaborn')


# In[3]:


print('hell')


# In[2]:


get_ipython().system(' pip install scikit-learn')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


laptop_data = pd.read_csv('datasets/laptops.csv', encoding='ISO-8859-1', index_col=0)


# In[5]:


laptop_data.head()


# In[ ]:





# In[6]:


laptop_data.shape


# In[7]:


laptop_data.drop(['Product','ScreenResolution','Cpu','Memory','Gpu','Weight'], inplace=True, axis=1)


# In[9]:


laptop_data.sample(10)


# In[12]:


laptop_data['TypeName'].unique()


# In[13]:


laptop_data.TypeName.value_counts()


# In[15]:


plt.figure(figsize=(12,8))
laptop_data['Company'].value_counts().plot(kind='bar')
plt.title('Laptops by company', fontsize=15)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# In[17]:


plt.figure(figsize=(12,8))
laptop_data[['Price_euros']].boxplot()


# In[19]:


laptop_data.boxplot(by='Company', column= ['Price_euros'], grid=False, figsize=(12,8))


# In[20]:


plt.figure(figsize=(12,8))
sns.swarmplot(x='TypeName', y='Price_euros', data=laptop_data)
plt.title('Price distribution by types', fontsize=15)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Price (in euros)', fontsize=12)
plt.show()


# In[24]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
laptop_data['TypeName'] = label_encoder.fit_transform(laptop_data['TypeName'])


# In[25]:


laptop_data.head()


# In[26]:


dummy_laptop_data=pd.get_dummies(laptop_data)
dummy_laptop_data.head()


# In[30]:


X = dummy_laptop_data.drop('Price_euros', axis=1)
Y = dummy_laptop_data['Price_euros']


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[32]:


X_train.shape, X_test.shape


# In[33]:


Y_train.shape, Y_test.shape


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)


# In[37]:


linear_regression.score(X_train, Y_train)


# In[38]:


y_pred = linear_regression.predict(X_test)
r2_score(Y_test, y_pred)


# In[39]:


plt.figure(figsize = (12,8))
plt.scatter(Y_test, y_pred)

plt.xlabel ('Acutal Value')
plt.ylabel ('Predicted Value')

plt.show()


# In[ ]:




