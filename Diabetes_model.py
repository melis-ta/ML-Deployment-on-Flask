#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


# In[2]:


#read the csv file and take a copy.
dataset = pd.read_csv(r'C:\Users\hmeli\OneDrive\Masaüstü\DataGlacierDataSets\diabetes.csv')
df=dataset.copy()


# In[3]:


#First look at the dataset
df.head()


# In[4]:


#informations about the dataset. 
df.info()


# In[5]:


#statistical informations of the data set.
df.describe().T


# ## Modelling
# 
# Creating the dependent and independent variables to apply Logistic regression model. 

# In[6]:


df["Outcome"].value_counts()


# In[7]:


X= df.iloc[:, :8]
y=df["Outcome"]


# In[8]:


#check the partition X.
X.head()


# In[9]:


#splitting the dataset as train set and test set by using sklearn.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state = 42)
loj=LogisticRegression(solver="liblinear")


# In[10]:


#fitting the model with training set.
loj_model= loj.fit(X_train,y_train)
loj_model


# In[11]:


#constant and coefficients of logistic regression model.
loj_model.intercept_


# In[12]:


loj_model.coef_


# In[13]:


#computed the predicted values with test test.
y_pred=loj_model.predict(X_test)


# In[14]:


#accuracy score of our model.
accuracy_score(y_test, y_pred)


# In[15]:


print(classification_report(y_test, y_pred))


# In[16]:


loj_model.predict_proba(X_test)[:5, :3]


# In[17]:


cross_val_score(loj_model, X_test, y_test, cv=10)


# In[18]:


cross_val_score(loj_model, X_test, y_test, cv=10).mean()


# In[19]:


pickle.dump(loj_model, open('Diabetes_model.pkl','wb'))


# In[20]:


loj_model = pickle.load(open('Diabetes_model.pkl','rb'))


# In[21]:


print(loj_model.predict_proba([[2, 90, 65, 12, 90, 35, 5, 67]]))


# In[ ]:




