import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import feature_calculators as fc


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

def zero_crossings(df):
    diff_lwx = np.diff(df.loc[:,'leftWrist_x'].to_numpy())    
    zc_lwx = fc.number_crossing_m(diff_lwx,0);
    diff_rwx = np.diff(df.loc[:,'rightWrist_x'].to_numpy());
    zc_rwx = fc.number_crossing_m(diff_rwx,0);
    diff_lwy = np.diff(df.loc[:,'leftWrist_y'].to_numpy());
    zc_lwy = fc.number_crossing_m(diff_lwy,0);
    diff_rwy = np.diff(df.loc[:,'rightWrist_y'].to_numpy())
    zc_rwy = fc.number_crossing_m(diff_rwy,0);        
    l = []
    l = [zc_lwx, zc_rwx, zc_lwy, zc_rwy]
    return l

def get_features(df):
    l = zero_crossings(df)
    
    return pd.DataFrame([l])

def main():  # use your path

    buy_all_files = glob.glob("data/buy" + "/*.csv")
    feature_matrix = pd.DataFrame()

    #filename = r'D:\Graduation_Courses\Mobile Computing\Assignment_2\CSV\buy\BUY_1_BAKRE.csv'
    for filename in buy_all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_req_columns = reqcolumns(df)
        #df_clean = dataclean(df_req_columns)
        df_norm = universal_normalize(df_req_columns)
        df_features = get_features(df_norm[['leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']])
        feature_matrix = pd.concat([feature_matrix,df_features],ignore_index=True)

    print(feature_matrix)

if __name__ == "__main__":
    main()