import flask
import io
import string
import time
import os
import sys
import librosa
import numpy as np
from flask import Flask, jsonify, request 
import model.clip2frame
from model.test_model_preppipline import predict_curSong
from model.test_model_preppipline import feat_extract

sys.path.append(os.path.dirname(__file__))
from model.cnn import score_pred_only


mp3 = '/Users/zhaoyufei/Desktop/spotify_analysis/data/test.mp3'

tag_feature = predict_curSong(mp3)
mel_Feature = feat_extract(mp3)
score = score_pred_only(mel_Feature, tag_feature)
print(score)
