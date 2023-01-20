from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os, requests, json
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

app = Flask(__name__)

res = requests.get('https://ipinfo.io/')
data = res.json()
city = data['city']

location = data['loc'].split(',')
latitude = location[0]
longitude = location[1]




model = load_model('saved_model/water_soil_wetness.hdf5')
reg_model = load_model('saved_model/reg_water_soil_wetness1.hdf5')
knn = pickle.load(open('saved_model/crop_predection.sav','rb'))

# Enter your API key here
api_key = "5b2691c07d8a7d9776ffb21270ca7474"

# base_url variable to store url
base_url = "https://api.openweathermap.org/data/2.5/weather?"
complete_url = base_url + "lat=" + latitude + "&lon=" + longitude +  "&appid=" + api_key

# return response object
response = requests.get(complete_url)

#print(response)

x = response.json()

if x["cod"] != "404":

	# store the value of "main"
	# key in variable y
	y = x["main"]

	# store the value corresponding
	# to the "temp" key of y
	current_temperature =round((y["temp"]-273), 2)
	print(round(2.665, 2))

	# store the value corresponding
	# to the "pressure" key of y
	#current_pressure = y["pressure"]

	# store the value corresponding
	# to the "humidity" key of y
	current_humidity = y["humidity"]

	# store the value of "weather"
	# key in variable z
	z = x["weather"]

	# store the value corresponding
	# to the "description" key at
	# the 0th index of z                like clear sky
	weather_description = z[0]["description"]

	
	# print following values
	print(" Temperature(C) = " +
					str(current_temperature) +
		"\n Humidity(%) = " +
					str(current_humidity))

else:
	print("City Not Found ")


'''
Current Temperature Variable name: current_temperature
Current Humidity Variable name: current_humidity
'''

model.make_predict_function()

def predict_label(img_path):
    img = load_img(img_path, target_size=(200,200))
    size = (200,200)    
    image = ImageOps.fit(img, size, Image.ANTIALIAS)
    x = np.asarray(image)
    #x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    a=np.argmax(model.predict(img_data), axis=1)
    if(a==0):
        return "Dry"
    elif(a==1):
        return "Extreamly Wet"
    else:
        return "wet"

def predict_moisture(img_path):
	img = load_img(img_path, target_size=(800,800))
	
	image = ImageOps.fit(img, (800, 800), Image.ANTIALIAS)
	x = np.asarray(img)
	x=np.expand_dims(x, axis=0)
	img_data=preprocess_input(x)
	a=reg_model.predict(img_data)
	return a[0][0]

def get_crop_pridiction():
	#input_data = (21.770462, 80.319644, 226.655537)
	
	input_data_as_numpy_array = np.asarray(input_data)

	input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
	std_data = scaler.transform(input_data_reshaped)

	prediction = knn.predict(std_data)
	print(prediction)

	return prediction


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
	return render_template("layout.html", content='predict')

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/dash")
def dashboard():
	#get_crop_pridiction()
	return render_template("layout.html", content='dashboard')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		soil_class = predict_label(img_path)
		soil_moisture = predict_moisture(img_path)
		print(soil_moisture)

	return render_template("Layout.html", prediction = soil_class, img_path = img_path, moisture=soil_moisture, content='predict', temp = current_temperature,  hum=current_humidity)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)