from flask import Flask, request, jsonify
import numpy as np
import pickle
from PIL import Image
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import cloudinary.uploader
import urllib.request

with open('./LR_model.pkl', 'rb') as modelFile:
    prediction_model = pickle.load(modelFile)

def binarization(img):
  (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  return np.invert(im_bw)

def crop_image(img,tol=0):
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def resize_image(img):
  img = cv2.resize(img, (224, 224))
  return img

base_model = VGG16(weights="imagenet", include_top = False, input_shape=(224,224,3))
for layer in base_model.layers:
  layer.trainable = False

model = pickle.load(open('./LR_model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
  file_to_upload = request.files['image']

  cloudinary.config(cloud_name = 'shiftnow', api_key='628486893553816', 
    api_secret='64qur2_WCzHFF9O6uiApCRo7klU')

  if file_to_upload:
    upload_result = cloudinary.uploader.upload(file_to_upload)
    app.logger.info(upload_result)


  # urllib.request.urlretrieve(
  # upload_result['url'],
  #  "gfg.png")
  # img = Image.open("gfg.png")
  # img = img.convert('L')   # convert to greyscale 
  # img.save('./output.png')

  resp = urllib.request.urlopen(upload_result['url'])
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

  #img = cv2.imread("./output.png",cv2.IMREAD_GRAYSCALE)
  binarized_img = binarization(img)
  cropped_img = crop_image(binarized_img)
  resized_img = resize_image(cropped_img)

  stacked_img = np.dstack((resized_img, resized_img, resized_img))
  input_img = np.expand_dims(stacked_img, axis=0)
  feature_matrix = base_model.predict(input_img)
  flattened_matrix = feature_matrix.reshape(-1)

  predicted_class = model.predict([flattened_matrix])
 
  response={
      "body":{"class": int(predicted_class[0])},
      "headers":{
        "Access-Control-Allow-Origin": '*'
      }
  }
  return jsonify(response)


@app.route('/')
def home():
  return "<h1>Hello</h1>"

if __name__ == '__main__':
  app.run()