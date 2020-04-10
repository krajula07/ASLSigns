import pandas as pd
#import numpy as np
import glob
from config import *


def reqcolumns(df):
    req_columns = ['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y','rightEye_x', 'rightEye_y',
               'leftShoulder_x', 'leftShoulder_y','rightShoulder_x', 'rightShoulder_y',
               'leftElbow_x','leftElbow_y', 'rightElbow_x', 'rightElbow_y',
               'leftWrist_x','leftWrist_y', 'rightWrist_x', 'rightWrist_y']
    return df[req_columns]

'''
def dataclean(df):
    count = df.isna().sum(axis=1)
    print(count)
    l = len(df)
    my_list = []
    for i in range(l):
        if(count[i] > 0.15*l):
            my_list.append(i)      
    
    df.drop(my_list, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.interpolate(method='linear',inplace=True,axis=1)
#        df.interpolate(method='quadratic',order=2, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
'''    
def universal_normalize(df):
    df['leftWrist_x'] = df.
    
def main(): # use your path
    
    buy_all_files = glob.glob(buy_path + "/*.csv")

    #filename = r'D:\Graduation_Courses\Mobile Computing\Assignment_2\CSV\buy\BUY_1_BAKRE.csv'
    for filename in buy_all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_req_columns = reqcolumns(df)
        #df_clean = dataclean(df_req_columns)
        df_norm = unviersal_normalize(df_req_columns)
    #df_features = get_features(df_norm)
    #pd.append([feature_matrix,df_features])
        
        #print(df_req_columns)

if __name__ == "__main__":
    main()