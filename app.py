from flask import Flask, render_template, request
import imageio
import numpy as np
from keras.models import load_model
import re
import sys
import os
import base64 as base_decoder
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
import time


sys.path.append(os.path.abspath('./model'))



app = Flask(__name__)


global model, graph
model = load_model('best_model.hdf5')
graph = tf.get_default_graph()
print(model.summary())


#decoding an image from base64 into raw representation
def convertImage(imgData):
	imgstr = base_decoder.b64decode(imgData)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr)


# define a route to hello_world function
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData[22:])
	x = imread('output.png', mode='L')
	x = np.invert(x)
	x = imresize(x, (28, 28)) / 255
	x = np.reshape(x, newshape=(1, 28, 28, 1))

	with graph.as_default():
		start = time.time()
		response = model.predict(x)
		digit = np.argmax(response[0])
		end = time.time()
		return str(digit) + ' and Execution Time:' + str(end - start)[:5] + ' seconds'




# Run the app on http://localhost:8085
if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='127.0.0.1', port=port)
