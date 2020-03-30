from flask_pymongo import PyMongo
from flask import Flask

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://devtotti:jankulovski@newclustera-shard-00-00-c85ej.mongodb.net:27017,newclustera-shard-00-01-c85ej.mongodb.net:27017,newclustera-shard-00-02-c85ej.mongodb.net:27017/blackpod?ssl=true&replicaSet=NewClusterA-shard-0&authSource=admin&retryWrites=true&w=majority"
mongo = PyMongo(app)




def getCloudWeather(month, year):
	db = mongo.db.weather

	for field in db.find():

		if ((str(month) == str(field['month'])) and (str(year) == str(field['year']))):
			temp_hi = str(field['temp_hi'])
			temp_lo = str(field['temp_lo'])
			rainfall = str(field['rainfall'])
			
			print(temp_hi, temp_lo, rainfall)
			return (temp_hi, temp_lo, rainfall)


#getCloudWeather("June", 2020)