# import libraries
from flask import Flask,  request
from flask_cors import CORS
import werkzeug
import json
from tensorflow import keras
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

# Initialize the flask App
app = Flask(__name__)
CORS(app)

#Load Model
emo_model = keras.models.load_model('models/facial_emotion_recognition_model.h5')

#Endpoint Function
@app.route('/getcatemo', methods=['GET', 'POST'])
def emotional_recommend():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('Upload/' + filename)

    img_path='Upload/' + filename
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dimension
    img = np.expand_dims(img, axis=0)  # add batch dimension

    result = emo_model.predict(img)
    le = LabelEncoder()
    le.fit(["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
    prediction_label = le.inverse_transform([result.argmax()])[0]
    prediction_label
    return_value = ' { "emotion" : "' + str(prediction_label) + '" }'

    print(return_value)
    return json.loads(return_value)

if __name__ == "__main__":
    app.run(debug=True)
