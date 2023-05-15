#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[2]:


df=pd.read_csv("C:\\Users\\Avijit\\Downloads\\wine.csv")


# In[3]:


df.head()


# In[4]:


df.corr()


# In[5]:


y=df['quality']


# In[6]:


#x=df[df.columns[1:12]]
x=df


# In[7]:


#x['acidity']=x['fixed acidity']+x['volatile acidity']
#x['dioxide']=x['total sulfur dioxide']-x['free sulfur dioxide']
x


# In[8]:


sns.heatmap(x.corr(), annot=True)


# In[9]:


#x=x[['sulphates','alcohol','dioxide']]
x=x.drop('quality',axis='columns')


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101) 


# In[11]:


from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(X_train, y_train)


# In[12]:


print(lm.intercept_)


# In[13]:


coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
coeff_df


# In[14]:


predictions = lm.predict(X_test)  
lm.score(X_train, y_train)
#plt.scatter(y_test,predictions)


# In[15]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

