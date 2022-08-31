import os
import sys
import base64
import time
import urllib.request
import datetime
import logging


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Metrics
from prometheus_flask_exporter import PrometheusMetrics

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
import cv2
import werkzeug
from PIL import Image 
from numpy import asarray 


logging.basicConfig(level=logging.INFO)
logging.info("Setting LOGLEVEL to INFO")


# Declare a flask app
app = Flask(__name__)
api = Api(app)
metrics = PrometheusMetrics(app)

metrics.info("app_info", "App Info, this can be anything you want", version="1.0.0")

# Model saved with Keras model.save()
MODEL_PATH = './models/datty.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Check http://127.0.0.1:5000/...')


def model_predict(img, model):
    
    imag = cv2.imread("./uploads/image.png")
    
    imag = cv2.resize(imag, (224, 224), interpolation = cv2.INTER_LINEAR)
    imag = imag / 255.
    imag = imag.reshape(1,224,224,3) 
    
    # Get Prediction
    reslist = []    
    prt = (model.predict(imag, batch_size=1)) 
    reslist.append(max(prt[0]))
    pty = np.argmax(model.predict(imag, 1, verbose = 0), axis=1)
    reslist.append(pty[0])

    # print (reslist)
    return reslist


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    # Health Check page
    return "OK",200


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        imwhole = np.array(img)

        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(imwhole, model)

        if preds[1]==1:
            result = str("NORMAL")            
        else:
            result = str("COVID") 

        return jsonify(result=result)

    return None

class Prediction_API(Resource):

    def post(self):
        t0 = time.process_time()
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['file']
        image_file.save("./uploads/image.png")

        imwholeg = np.array(image_file)
        preds = model_predict(imwholeg, model)
        if preds[1]==1:
            result = str("NORMAL")           
        else:
            result = str("COVID")

        t1 = time.process_time()
        total = t1-t0

        retMap = {
            "API vesion": "1",
            "Response time": str(total) + " seconds",
            "disclaimer": "This API does not claim any medical correctness for the rendered results",
            "result": {
                "Message" : "Image successfully received",
                "confidence": str(preds[0]),
                "predicted_label": result
            }
        }

        return jsonify(retMap)

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status':404,
        'message':'Not Found' + request.url
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp




api.add_resource(Prediction_API, "/api")


if __name__ == '__main__':
    # Serve the app with gevent
    port = int(os.environ.get('PORT', 5000))
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
