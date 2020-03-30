# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:16:48 2020

@author: dolapo
"""
#import libraries
import timeit 
import pandas as pd 
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt

#execution time start 
#start = timeit.default_timer() 

#loading the dataset
dataset=pd.read_excel('db1.xlsx')
dataset.head()

#preprocessing stage
index=['Date']
dataset=dataset.set_index(index)
drops=['Rainfall','TempMin','TARGET']
dataset.drop(drops,axis=1,inplace=True)

#building the model
stepwise_model=auto_arima(dataset, start_p=1, start_q=1,
                          max_p=5, max_q=5, m=12,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True)

#print(stepwise_model.aic())

#train, test split 
train=dataset.loc['1988-01-01':'2010-12-01']
test=dataset.loc['2011-01-01':]

#the train data is fit into the model
stepwise_model.fit(train)
#prediction
forecast=stepwise_model.predict(n_periods=len(test))
forecast=pd.DataFrame(forecast,index=test.index,columns=['Prediction'])
#execution time stop 
#stop = timeit.default_timer() 

#print('Time: ', stop - start)  #calculate 

#plotting 
pd.concat([dataset,forecast],axis=1).plot()
pd.concat([test,forecast],axis=1).plot()
#confusion matrix
new_list=[]
for item in forecast:
    if item<=30:
        new_list.append(1)
    else:
        new_list.append(0)
#print(new_list)
tests=test['TempMax']
new_list1=[]
for item in tests:
    if item<=30:
        new_list1.append(1)
    else:
        new_list1.append(0)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(new_list,new_list1)

#Root mean square
from sklearn.metrics import mean_squared_error
from math import sqrt
rmss=sqrt(mean_squared_error(new_list1,new_list))
       
#plot the prediction 
plt.plot(train, label='Train')
plt.plot(test, label='test')
plt.plot(forecast, label='Prediction')
plt.show()
#precision,recll,accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(new_list,new_list1)
precision = precision_score(new_list1,new_list)
recall = recall_score(new_list,new_list1)