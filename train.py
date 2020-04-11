import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import feature_calculators as fc
from decimal import Decimal as dc
import math

def reqcolumns(df):
    req_columns = ['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y',
                   'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x', 'rightShoulder_y',
                   'leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y',
                   'leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']
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
    df.loc[:,'leftWrist_x'] = abs((df.loc[:,'leftWrist_x']-df.loc[:,'nose_x'])/(df.loc[:,'leftEye_x']-df.loc[:,'rightEye_x']))
    df.loc[:,'rightWrist_x'] = abs((df.loc[:,'rightWrist_x']-df.loc[:,'nose_x'])/(df.loc[:,'leftEye_x']-df.loc[:,'rightEye_x']))
    df.loc[:,'leftWrist_y'] = abs((df.loc[:,'leftWrist_y']-df.loc[:,'nose_y'])/(df.loc[:,'nose_y']-df.loc[:,'leftShoulder_y']))
    df.loc[:,'rightWrist_y'] = abs((df.loc[:,'rightWrist_y']-df.loc[:,'nose_y'])/(df.loc[:,'nose_y']-df.loc[:,'rightShoulder_y']))
    return df

def split_leftwrist_x(df):
    return df.loc[:,'leftWrist_x'].to_numpy()

def split_leftwrist_y(df):
    return df.loc[:,'leftWrist_y'].to_numpy()

def split_rightwrist_x(df):
    return df.loc[:,'rightWrist_x'].to_numpy()

def split_rightwrist_y(df):
    return df.loc[:,'rightWrist_y'].to_numpy()



def min_max_points(df):
    diff_lwx = np.diff(split_leftwrist_x(df))
    diff_rwx = np.diff(split_rightwrist_x(df))
    diff_lwy = np.diff(split_leftwrist_y(df))
    diff_rwy = np.diff(split_rightwrist_y(df))
    
    x = [diff_lwx, diff_lwy, diff_rwx, diff_rwy]
    min_max = []
    
    for i in x:
        l = len(i)
        j = 0
        while j<l: 
            min1 = fc.minimum(i[j:j+math.ceil(l/3)])
            max1 = fc.minimum(i[j:j+math.ceil(l/3)])
            dev =  dc(max1) - dc(min1)
            min_max.append(dev)  
            j = j+math.ceil(l/3)
    return pd.DataFrame(min_max).transpose()
    
def zero_crossings(df):
    diff_lwx = np.diff(split_leftwrist_x(df))
    diff_rwx = np.diff(split_rightwrist_x(df))
    diff_lwy = np.diff(split_leftwrist_y(df))  
    diff_rwy = np.diff(split_rightwrist_y(df))
    
    x = [diff_lwx, diff_lwy, diff_rwx, diff_rwy]
    zc = []
    for i in x:
        l = len(i)
        j = 0
        while j<l:
            zc_i = fc.number_crossing_m(i[j:j+math.ceil(l/3)],0)
            zc.append(zc_i)
            j = j+math.ceil(l/3)
    return pd.DataFrame(zc).transpose()


def get_features(df):
    l = zero_crossings(df)
    m_m = min_max_points(df)
    return pd.concat([l,m_m],axis=1)

  # use your path
buy_all_files = glob.glob("data/buy" + "/*.csv")
feature_matrix = pd.DataFrame()
    
#filename = r'D:\Graduation_Courses\Mobile Computing\Assignment_2\CSV\buy\BUY_1_BAKRE.csv'
for filename in buy_all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df_req_columns = reqcolumns(df)
    df_norm = universal_normalize(df_req_columns)
    
    df_features = get_features(df_norm[['leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']])
    
    feature_matrix = pd.concat([feature_matrix,df_features],ignore_index=True)
    #print(feature_matrix)
