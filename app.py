import flask
import io
import string
import time
import os
import librosa
import numpy as np
from flask import Flask, jsonify, request 
import model.clip2frame
from model.test_model_preppipline import predict_curSong
from model.test_model_preppipline import feat_extract
from model.cnn import score_pred_only

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_mp3():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The mp3 file doesn't exist"
    
    # get file
    file = request.files.get('file')

    # temporarily save file 
    store_dir = os.path.join(os.path.dirname(__file__), 'filestorage')
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    file_name = os.path.join(store_dir, 'test.mp3')
    file.save(file_name)

    # predict
    mel_Feature = feat_extract(file_name)
    tag_feature = predict_curSong(file_name)
    score = score_pred_only(mel_Feature, tag_feature).tolist()

    # remove file
    os.remove(file_name)

    # Return on a JSON format
    return jsonify(score)

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')