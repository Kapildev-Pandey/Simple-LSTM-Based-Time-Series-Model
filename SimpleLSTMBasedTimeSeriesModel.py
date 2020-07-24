#!/usr/bin/env python
# coding: utf-8

# ## Dataset:
# The dataset I used here is from Stock price data of the fifty stocks in NIFTY-50 index from [NSE India](https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=KOTAKBANK.csv). You can download and can start.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


kota=pd.read_csv('/_Give_your_data_path_/datasets_423609_1296539_KOTAKBANK.csv')


# In[3]:


data=kota.copy()                # Copying into data variabe.


# ### Let see our top five values.

# In[4]:


data.head()


# ### Let see our bottom five values.

# In[5]:


data.tail()


# As we can see there are lots of columns or features of but for a moment we will see only only one variale called **'Close'**, which is our feature and try to make a model.
# 
# ### Let see the size of the dataset.

# In[6]:


data.shape


# ### This our features.

# In[7]:


data.columns


# ### Converting dataframe value into array.

# In[8]:


data=data['Close'].values


# In[9]:


data


# ### Let's plot.

# In[10]:


plt.figure(figsize=(16,8))
plt.plot(kota.index,kota['Close'])
plt.title('Closing values')
plt.xlabel('Treding period year 2000-2020',fontsize=15)
plt.ylabel('Closing values over different year',fontsize=16)
plt.legend(['Closing value Trend'])
plt.grid()
plt.show()


# ### We will ues MinMaxScaler to normalize it.

# In[11]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

scalar=MinMaxScaler(feature_range=(0,1))
dataset=scalar.fit_transform(data.reshape(data.shape[0],1))
dataset


# ### Let see our dataset size.

# In[12]:


dataset.shape


# In[13]:


dataset=dataset.reshape(data.shape[0])


# ### This is our function which will split the data into target and features.

# In[14]:


def split_set(data,time_steps):
    split_X=[]
    split_y=[]
    split=[[data[j+i] for j in range(time_steps)] for i in range(len(data)-time_steps)]
    split_X=split
    
    for i in range(len(data)-time_steps):
        split_y.append([data[i+time_steps]])
    return np.array(split_X),np.array(split_y)


# ### So here our timeP_steps=1. We can also change it by our convience.

# Here we are spliting the dataset.

# In[15]:


time_steps=1
split_X,split_y=split_set(dataset,time_steps)
split_X


# ### Here we splitting the dataset into train and test. 80 percent for train and 20 percent for test dataset.

# In[16]:


train_size=int(len(split_X)*0.80)
test_size=len(split_X)-train_size

print('Train-size=',train_size)
print('Test-size=',test_size)


# In[17]:


train_X=split_X[0:train_size,]
train_y=split_y[0:train_size,]


test_X=split_X[train_size:len(dataset),]
test_y=split_y[train_size:len(dataset),]


# ### Let's see our samples shape

# In[18]:


print(test_X.shape)
print(test_y.shape)


# ### Reshaping the train_X and test_X to feed into the model and predict.

# In[19]:


train_X=train_X.reshape(train_X.shape[0],time_steps,1)
test_X=test_X.reshape(test_X.shape[0],time_steps,1)


# In[20]:


print(train_X.shape)


# ### This is our functional keras model

# In[21]:


import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Input
from tensorflow.keras.models import Model

data_input=Input(shape=(time_steps,train_X.shape[2]))
lstm1=LSTM(50,return_sequences=True)(data_input)
lstm2=LSTM(30)(lstm1)
dense_output=Dense(20)(lstm2)
output=Dense(1)(dense_output)
model=Model(data_input,output)
model.summary()


# ### Lets fit the training sample into model

# In[22]:


model.compile(loss='mse', optimizer='adam')
model.fit(train_X,train_y,epochs=10,batch_size=10)


# ### So let's see our model prediction on test_X

# In[23]:


prediction=model.predict(test_X)
prediction


# ### Note that we want our original data so we will inverse_transform our prevoius transformed dataset.

# In[24]:


fit_data=scalar.fit(data.reshape(data.shape[0],1))
#fit_data.inverse_transform(np.array(prediction))


# In[25]:


original_target_feature=fit_data.inverse_transform(split_y)   #from split_y
original_prediction_test=fit_data.inverse_transform(prediction)


# In[26]:


train=original_target_feature.reshape(split_y.shape[0]).copy()        
train[train_size:len(split_y),]=original_prediction_test.reshape(prediction.shape[0])

plt.figure(figsize=(16,8))
plt.plot(train)
plt.plot(original_target_feature.reshape(split_y.shape[0]))
plt.title('Closing values')
plt.xlabel('Treding period year 2000-2020',fontsize=15)
plt.ylabel('Closing values over different year',fontsize=16)
plt.grid()
plt.legend(['Prediction','Original closing value'])
plt.show()


# As we can seee it seems to be good prediction.That's great. Now we zoom it on test dataset part. Which will give us better idea and visulization.

# ### Zooming on test_data 

# In[27]:


for i in [10,100,200,400,700]:
    plt.figure(figsize=(16,8))
    plt.plot(train[train_size+i:len(split_y)])
    plt.plot(original_target_feature.reshape(split_y.shape[0])[train_size+i:len(split_y)])
    plt.title('Closing values')
    plt.xlabel('Treding period year 2000-2020',fontsize=15)
    plt.ylabel('Closing values over different year',fontsize=16)
    plt.grid()
    plt.legend(['Prediction','Original closing value'])
    plt.show()

