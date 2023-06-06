from flask import Flask, request
import torch
import os
import json
import numpy as np
import pandas as pd
from utils.preprocessing import get_spectrogram,\
                               get_spectrogram_and_label_id,\
                               get_log_mel_features_and_label_id,\
                               get_mfcc_and_label_id

def predict(filename):

    log_mel_features = log_mel_feature_extraction(audio)


    raise NotImplementedError

    
app = Flask(__name__)
@app.route('/<filename>', methods=['GET', 'POST'])
def predict_api(filename):
    if request.is_json: 
        content = request.get_json(force=True)
        name = content['filename']
        json_text = predict(filename)
        return json_text
    else:
        print ("request is not json")
        # convert to json
        name = content['filename']
        json_text = predict(filename)
        return json_text
    
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='checkpoint', help='path to directory with model and vocabularies')
args = parser.parse_args()

if __name__ == '__main__':
    MODEL_PATH = args.model_path
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)
    # app.run(host='127.0.0.1', port=5000)