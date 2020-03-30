# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 07:01:47 2020

@author: dolapo
"""
#import libraries

from flask import Flask, jsonify, request
from fetch_data_from_cloud import *



import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import auto_arima



import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle



@app.route('/', methods = ['POST'])
def get_prediction():
	month = request.get_json()['month']
	year = request.get_json()['year']

	temp_hi, temp_lo, rainfall = getCloudWeather(month, year)
	temp_hi = float(temp_hi)
	temp_lo = float(temp_lo)
	rainfall = float(rainfall)

	prediction = mainScript(temp_hi, temp_lo, rainfall)

	response = {"prediction":prediction[0]}


	return jsonify(response)






def testScript(temp_hi, temp_lo, rainfall):
	#load the dataset
	dataset = pd.read_excel('db1.xlsx')
	#set date as index
	dataset=dataset.set_index('Date')
	#calculate the p,d,q
	stepwise_model=auto_arima(dataset['TempMax'], start_p=1, start_q=1,
                              	max_p=5, max_q=5, m=12,
                              	start_P=0, seasonal=True,
                              	d=1, D=1, trace=True, error_action='ignore',
                              	suppress_warnings=True, stepwise=True)
	#fit the temp to the model
	stepwise_model.fit(dataset['TempMax'])
	#Predict 
	forecast=stepwise_model.predict(n_periods=len(dataset['TempMax']))
	#prediction output into dataframe
	forecast = pd.DataFrame(forecast,columns=['Pred'])
	#new data created and dataset date set as index
	new_data=dataset.join(forecast.set_index(dataset.index))

	#data split into train and test set
	inputs=['Rainfall','TempMin','Pred']
	outputs=['TARGET']
	X_train,X_test,y_train,y_test=train_test_split(new_data[inputs],new_data[outputs],test_size=0.25,random_state=0)
	#Decision tree classifier called 
	classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
	#X_train and y_train fit to the classifier
	classifier.fit(X_train,y_train)

	#prediction
	y_pred=classifier.predict(X_test)

	#visualization
	forecast=pd.DataFrame(y_pred,index=y_test.index,columns=['Prediction'])
	pd.concat([y_test,forecast],axis=1).plot()
	#confusion matrix
	
	cm = confusion_matrix(new_list,new_list1)
	#creating the model
	
	filename='model1.pkl'
	with open(filename,'wb')as file:
		pickle.dump(classifier,file)

	with open('model1.pkl','rb') as f:
		loaded_model = pickle.load(f)
	
	my_data=pd.DataFrame({'Rainfall':[rainfall],'TempMin':[temp_lo],'TempMax':[temp_hi]})
	prediction = loaded_model.predict(my_data)
	print(prediction)

	return prediction




def mainScript(temp_hi, temp_lo, rainfall):
	filename = 'model1.pkl'

	with open (filename, 'rb') as modelFile:
		model = pickle.load(modelFile)

	dataframe = pd.DataFrame({'Rainfall':[rainfall],'TempMin':[temp_lo],'TempMax':[temp_hi]})
	prediction = model.predict(dataframe)
	print(prediction[0])

	return prediction





if __name__ == "__main__":
    app.run(debug=True, port=5000)





