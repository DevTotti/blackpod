# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:14:23 2020

@author: dolapo
"""

#importing libraries
import timeit 
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import pandas as pd

start = timeit.default_timer()
#loading the dataset
dataset=pd.read_excel('db1.xlsx')
output={'Blackpod':1,'No Blackpod':0}
dataset=dataset.replace({'TARGET':output})

#preprocessing stage
index=['Date']
dataset=dataset.set_index(index)
#drops=['Rainfall','TempMin','TARGET']
data=dataset['TempMax']
#train = dataset[:int(0.7*(len(dataset)))]
#test = dataset[int(0.7*(len(dataset))):]
stepwise_model=auto_arima(data, start_p=1, start_q=1,
                          max_p=5, max_q=5, m=12,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True)

train=data.loc['1988-01-01':'2010-12-01']
test=data.loc['2011-01-01':]


stepwise_model.fit(data)
forecast=stepwise_model.predict(n_periods=len(data))
forecast = pd.DataFrame(forecast,columns=['Prediction'])
new_data=dataset.join(forecast.set_index(dataset.index))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
inputs=['Rainfall','TempMin','Prediction']
outputs=['TARGET']
X_train,X_test,y_train,y_test=train_test_split(new_data[inputs],new_data[outputs],test_size=0.25,random_state=0)
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#from sklearn.ensemble import VotingClassifier 
#ens=VotingClassifier(estimators[('arima',),('Decision',)],voting='hard')
#ens.fit()
#ens.predict()

#mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#calculate MAPE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
#decision
inputs=['Date']
outputs=['TempMax']

X=dataset[inputs]
y=dataset[outputs]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
predict=regressor.predict(X_test)
y_tests=y_test['TempMax']

#confusion matrix
#pred=forecast['Prediction']
new_list=[]
for item in forecast:
    if item<=30:
        new_list.append('1')
    else:
        new_list.append('0')
#print(new_list)
#tests=test['TempMax']
new_list1=[]
for item in test:
    if item<=30:
        new_list1.append('1')
    else:
        new_list1.append('0')
        
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(new_list,new_list1)