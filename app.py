import flask
import io
import string
import time
import os
import librosa
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect, url_for
import model.clip2frame
from model.test_model_preppipline import predict_curSong
from model.test_model_preppipline import feat_extract
from model.cnn import score_pred_only
from model.test_model_preppipline import compute_percentile

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])
def upload_file():
    uploaded_file = request.files['file']
    # temporarily save file 
    store_dir = os.path.join(os.path.dirname(__file__), 'filestorage')
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    if uploaded_file.filename != '':
        file_name = os.path.join(store_dir, 'test.mp3')
        uploaded_file.save(file_name)
    
    # predict
    mel_Feature = feat_extract(file_name)
    tag_feature = predict_curSong(file_name)
    score = score_pred_only(mel_Feature, tag_feature).tolist()[0]
    pop = round(score[0],0)
    pop_percentile = round(compute_percentile(pop),2)

    # remove file
    os.remove(file_name)

    #return jsonify(score)
    return render_template('result.html', **locals())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')