from flask_pymongo import PyMongo
from flask import Flask
from bson.objectid import ObjectId
from datetime import datetime
import json


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://devtotti:jankulovski@newclustera-shard-00-00-c85ej.mongodb.net:27017,newclustera-shard-00-01-c85ej.mongodb.net:27017,newclustera-shard-00-02-c85ej.mongodb.net:27017/blackpod?ssl=true&replicaSet=NewClusterA-shard-0&authSource=admin&retryWrites=true&w=majority"
mongo = PyMongo(app)


def toCloud():
	db = mongo.db.weather


	with open ('database.json') as database:
		response = json.loads(database.read())

	datalenght = len(response)

	if response:
		for data in response:
			print (data)
			saves = db.insert_one(data)



def saveInfo(name, age, state, phone, add, farm):
	db = mongo.db.users

	data = {"name":name, "age": age, "state of residence":state, "phone number":phone, "address": add, "farm location": farm}

	try:
		save = db.insert_one(data)


		print("Data saved successfully")

		response = "Success! "


	except Exception as error:
		
		print("Error saving data into database")

		response = "Failed!"


	return response



