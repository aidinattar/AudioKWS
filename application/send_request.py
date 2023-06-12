import requests
import argparse
import json
import pandas as pd
import re

parser = argparse.ArgumentParser(description='send request to API')
parser.add_argument('--filename', type=str,help='filename to classify', default='nicole zattarin')
parser.add_argument('--real_label', type=str,help='real label', default='nicole zattarin')

args = parser.parse_args()

if __name__ == '__main__':

    dict_request = {'filename': args.filename, }
    host = 'http://127.0.0.1:5000/'
    res = requests.post(host, json=dict_request)

    # html to json
    text = res.text
    print ("filename: ", args.filename)
    print ("real label: ", args.real_label)
    try:
        text = dict(json.loads(text))
        df = pd.DataFrame.from_dict(text, orient='index', columns=['confidence'])
        print (df)
    except:
        # rmeove html tags, i.e. everything between <>
        text = re.sub('<[^<]+?>', '', text)
        print (text)

        
        
