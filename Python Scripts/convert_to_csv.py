import glob
import numpy as np
import pandas as pd
from numpy import genfromtxt


import random
import csv
import json
import os


# function to convert JSON input
def convert(input_file):
    headers = True
    with open(input_file) as f:
        reads = f.read()
        j = json.loads(reads)
        if os.path.exists('converted.csv'):
            os.remove('converted.csv')
        csv_file = open('converted.csv', 'a')
        #print(j)
        header = ['Frames#', 'score_overall']
        for i in range(len(j)):
            row = [i, j[i]["score"]]

            for k in j[i]['keypoints']:
                col = ''
                col = col + str(k['part'])
                header.append(str(k['part']) + "_score")
                row.append(k["score"])
                for l in k['position']:
                    header.append(col + "_" + l)
                    row.append(k["position"][l])

            writer = csv.writer(csv_file)
            if headers:
                #print(header)
                writer.writerow(header)
                headers = False
            writer.writerow(row)
