#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__)
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv('Tata-steel.csv')
df.head(10)


# In[2]:


# To change date in the accepted format
df['Date']=pd.to_datetime(df.Date)
df.head(10)


# In[3]:


df.info()


# In[4]:


# To check whether we have any missing Data 
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[37]:


#EDA
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='High Price',y='Low Price',data=df,color='blue')


# In[28]:


df['Open Price'].plot.hist()


# In[29]:


plt.style.use('ggplot')


# In[39]:


df['Open Price'].plot.hist(alpha=0.5,bins=25,color='blue')


# In[6]:


df[['Open Price','Close Price','High Price','Close Price']].plot(kind='box')


# In[7]:


layout=go.Layout(
    title='Stock price of Tata Steel',
    xaxis=dict(
        title='Date'
       
    ),
    yaxis=dict(
        title='Price'
        
    )
)
df1=[{'x':df['Date'],'y':df['Close Price']}]
plot=go.Figure(data=df1,layout=layout,)
iplot(plot)


# In[8]:


#PREDICTION BY A REGRESSION MODEL
#Building the regression MOdel
from sklearn.model_selection import train_test_split

#for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#For Model Evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[9]:


# Split the data into train and test sets
X=np.array(df.index).reshape(-1,1)
Y=df['Close Price']
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=1001)


# In[10]:



#feature Scaling
scaler= StandardScaler().fit(X_train)
from sklearn.linear_model import LinearRegression


# In[11]:


#creating a linear model
lm=LinearRegression()
lm.fit(X_train,Y_train) 


# In[12]:


#Plot Actual And predicted values for train dataset
trace0 = go.Scatter(
x = X_train.T[0],
y=Y_train,
mode='markers',
name='Actual'
)
trace1 = go.Scatter(
x = X_train.T[0],
y=lm.predict(X_train).T,
mode='lines',
name='Predicted'
)
df1=[trace0,trace1]
layout.xaxis.title.text='Day'
plot2 = go.Figure(data=df1,layout=layout)
iplot(plot2)
                


# In[13]:


scores =f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train,lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train,lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(scores)


# In[14]:


df1=df.reset_index()['Close Price']


# In[15]:


#ANOTHER LINEAR REGRESSION MODEL(Approach 2)

X=df[['Open Price','High Price','Low Price','No.of Shares']]
y=df['Close Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)
X_train.shape


# In[16]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[17]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[18]:


predictions = lm.predict( 
    X_test)


# In[35]:


plt.scatter(y_test,predictions,color='blue')
plt.xlabel('Y Test',fontsize=20,color='red')
plt.ylabel('Predicted Y',fontsize=20)
plt.plot


# In[40]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[41]:


scores =f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(y_train,lm.predict(X_train))}\t{r2_score(y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(y_train,lm.predict(X_train))}\t{mse(y_test,lm.predict(X_test))}
'''
print(scores)


# In[42]:


#NOW WE WOLL DO PRDCTION USING TENSORFLOW AND PREDICT THE PRICE FOR NEXT 10 DAYS

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[43]:


print (df1)


# In[44]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[45]:


training_size,test_size


# In[46]:


train_data


# In[47]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[48]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[49]:


print(X_test.shape), print(ytest.shape)


# In[50]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[51]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[52]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[58]:


model.summary()


# In[59]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[60]:


import tensorflow as tf


# In[61]:


tf.__version__


# In[62]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[63]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[64]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[69]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[105]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1),color='red')
plt.plot(trainPredictPlot,color='blue')
plt.plot(testPredictPlot,color='yellow')
plt.xlabel('Days',fontsize=18)
plt.ylabel('Price In rupees',fontsize=20)
plt.legend(["Original Price","train price","test price"],loc="upper left")


# In[76]:


len(test_data)


# In[77]:


x_input=test_data[420:].reshape(1,-1)
x_input.shape


# In[78]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[79]:


temp_input


# In[80]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[89]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[90]:


len(df1)


# In[99]:


plt.plot(day_new,scaler.inverse_transform(df1[1385:]),color='red')
plt.plot(day_pred,scaler.inverse_transform(lst_output),color='blue')
plt.xlabel('Days',fontsize=20,color='black')
plt.ylabel('Price In rupees',fontsize=20,color='black')

plt.legend(["Original Price","predicted price"],loc="lower right") #predicted rice for next 10 days


# In[96]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1400:],color='black')


# In[123]:


df3=scaler.inverse_transform(df3).tolist()


# In[100]:


plt.plot(df3,color='blue')
plt.xlabel('Days',fontsize=20)
plt.ylabel('Price In rupees',fontsize=20)


# In[ ]:




