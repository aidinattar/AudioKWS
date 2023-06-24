import requests
import argparse
import json
import pandas as pd
import re
import numpy as np

parser = argparse.ArgumentParser(description='send request to API')
parser.add_argument('--N', type=str,help='number of images to predict', default='1')

args = parser.parse_args()

if __name__ == '__main__':
    host = 'http://127.0.0.1:5000/'
    endpoint = args.N
    host = host + endpoint
    print ("host: ", host)
    res = requests.get(host,)

    # html to json
    text = res.text
    print ("n: ", args.N)
    # remove all the html tags and img

    text = re.sub('<img src="static/\d+.png" alt="img\d+" width="\d+" height="\d+">', '', text)
    labels = [i.split('Label: ')[1].split(', Predicted: ')[0].replace(' ', '') for i in text.split('<p>')[1:]]
    preds = [i.split('Label: ')[1].split(', Predicted: ')[1].replace(' ', '').replace('</p>', '') for i in text.split('<p>')[1:]]

    if args.N == '1':
        if labels == preds: 
            print ("Correct prediction for ", labels)
        else:
            print ("Incorrect prediction for ", labels)

    else:
        print ("labels: ", labels)
        print ("preds: ", preds)
        acc = sum(np.array(labels) == np.array(preds))/ len(labels)
        print (f"Accuracy: over {args.N} images is {acc}")


        
