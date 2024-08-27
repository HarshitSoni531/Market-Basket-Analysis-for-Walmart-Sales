#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[8]:


get_ipython().system('pip install apyori')


# In[9]:


df=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
df


# In[11]:


transactions=[]
for i in range(0,7502):
    transactions.append([str(df.values[i,j]) for j in range(0,20)])
        


# In[13]:


from apyori import apriori 
rules=apriori(transactions=transactions , min_support = 0.003 , min_confidence = 0.2,min_lift=3,min_length=2,max_length=2)


# In[17]:


results=list(rules)


# In[18]:


results


# In[19]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[20]:


resultsinDataFrame 


# In[21]:


resultsinDataFrame.nlargest(n=10 , columns = 'Lift')


# In[ ]:




