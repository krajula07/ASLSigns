import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
from tsfresh.feature_extraction import feature_calculators as fc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from decimal import Decimal as dc
import math
import scipy.fftpack as fft
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils

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
    return df.loc[:,'leftWrist_x']

def split_leftwrist_y(df):
    return df.loc[:,'leftWrist_y']

def split_rightwrist_x(df):
    return df.loc[:,'rightWrist_x']

def split_rightwrist_y(df):
    return df.loc[:,'rightWrist_y']



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

def FastFourierTransform(df):
    x = [split_leftwrist_x(df), split_leftwrist_y(df),split_rightwrist_x(df), split_rightwrist_y(df)]
    fft_amp_df = pd.DataFrame()
    for i in x:
        fft_output = fft.rfft(i, n=5, axis=0)
        fft_amp = np.absolute([fft_output])
        fft_amp_df = pd.concat([fft_amp_df,pd.DataFrame(fft_amp)],axis=1)
        
    return fft_amp_df


def get_features(df):
    l = zero_crossings(df)
    m_m = min_max_points(df)
    fft_features = FastFourierTransform(df)
    return pd.concat([l,m_m,fft_features],axis=1)

def PCA_fit(feature_matrix, dimension,sign):
    fm_after_ss = StandardScaler().fit_transform(feature_matrix)
    pca = PCA(n_components=dimension)   
    pca.fit(fm_after_ss)

    pca_file_name = sign+"_PCA.pickle"
    fname = open(pca_file_name, 'wb')
    pickle.dump(pca, fname)
    fname.close()
    return fm_after_ss

def DimensionalityReduction(fm,file):
    f = open(file,'rb')
    pca = pickle.load(f)
    f.close()
    return pca.transform(fm)

def SupportVectorMachine(x_train,x_test,y_train,y_test,sign):    
    svc=SVC(kernel='linear')
    svc.fit(x_train,y_train)
    y_pred = svc.predict(x_test)
    
    model_file_name = sign+"_svm.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(svc, filename)
    filename.close()
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy SupportVectorMachine : ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))
    
    return accuracy_score(y_test,y_pred)*100

def Logistic_Regression(x_train,x_test,y_train,y_test,sign):
    lr= LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    model_file_name = sign+"_lr.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(lr, filename)
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Logistic_Regression: ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))

# Logistic_Regression()

def Linear_Discriminant_Analysis(x_train,x_test,y_train,y_test,sign):
    lda= LinearDiscriminantAnalysis()
    lda.fit(x_train,y_train)
    y_pred = lda.predict(x_test)
    model_file_name = sign+"_lda.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(lda, filename)
    filename.close()
    
#    print(y_pred)
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Linear_Discriminant_Analysis: ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))
    
    
def RandomForest(x_train,x_test,y_train,y_test,sign):
    rf = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1, random_state=101, max_features=None)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    model_file_name = sign+"_random_forest.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(rf, filename)
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Random Forest : ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))
    
def GradientBoostClassifier(x_train,x_test,y_train,y_test,sign):
    gbc = GradientBoostingClassifier(n_estimators=1000)
    gbc.fit(x_train , y_train)
    y_pred = gbc.predict(x_test)
    model_file_name = sign+"_gradientboost.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(gbc, filename)
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Gradient Boost Classifier: ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))

def AdaBoost(x_train,x_test,y_train,y_test,sign):
    abc = AdaBoostClassifier()
    abc.fit(x_train , y_train)
    y_pred = abc.predict(x_test)
    model_file_name = sign+"_adaboost.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(abc, filename)
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy AdaBoost Classifier: ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))

def Decision_tree(x_train,x_test,y_train,y_test,sign):
    dtree = DecisionTreeClassifier(max_depth=10,min_samples_leaf=10)
    dtree.fit(x_train, y_train)
    y_pred = dtree.predict(x_test)
    model_file_name = sign+"_decision_tree.pkl"   
    filename = open(model_file_name, 'wb')
    pickle.dump(dtree, filename)
    filename.close()
    
#  print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Decision Tree: ",accuracy_score(y_test,y_pred)*100)
#   print("Report : ",classification_report(y_test, y_pred))

def KNN(x_train,x_test,y_train,y_test,sign):
    knn=KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=4, p=2,weights='uniform')
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    model_file_name = sign+"_knn.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(knn, filename)      
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy : ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))

def Quadratic_Discriminant_Analysis(x_train,x_test,y_train,y_test,sign):
    qda= QuadraticDiscriminantAnalysis()
    qda.fit(x_train,y_train)
    y_pred = qda.predict(x_test)
    model_file_name = sign+"_qda.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(qda, filename)
    filename.close()
    
#    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ( sign+" Accuracy Quadratic_Discriminant_Analysis: ",accuracy_score(y_test,y_pred)*100)
#    print("Report : ",classification_report(y_test, y_pred))
    
def NaiveBayes_Classifier(x_train,x_test,y_train,y_test,sign):
    nbc = GaussianNB()
    nbc.fit(x_train,y_train)
    y_pred = nbc.predict(x_test)
    model_file_name = sign+"_nbc.pkl"
    filename = open(model_file_name, 'wb')
    pickle.dump(nbc,filename)
    filename.close()
    
    print ( sign+" Accuracy Naive Bayes Classifier: ",accuracy_score(y_test,y_pred)*100)
    

#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(8, input_dim=5, activation='relu'))
#	model.add(Dense(6, activation='softmax'))
#	# Compile model
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
#    
#def neural_networks():
#    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
#    kfold = KFold(n_splits=10, shuffle=True)
#    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# use your path

'''
def train_model_sign(all_files, sign):
    feature_matrix = pd.DataFrame()
    
#filename = r'D:\Graduation_Courses\Mobile Computing\Assignment_2\CSV\buy\BUY_1_BAKRE.csv'
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_req_columns = reqcolumns(df)
        df_norm = universal_normalize(df_req_columns)
        
        df_features = get_features(df_norm[['leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']])
        
        feature_matrix = pd.concat([feature_matrix,df_features],ignore_index=True)
    #print(feature_matrix)
    PCA_fit(feature_matrix,5)
    updated_feature_matrix_sign = DimensionalityReduction(feature_matrix, sign_+"PCA.pickle")
    
    r, c = updated_feature_matrix_sign.shape
    updated_feature_matrix_not_sign = updated_feature_matrix_sign - np.random.rand(r,c)
        
    updated_feature_matrix_sign = pd.DataFrame(updated_feature_matrix_sign)
    updated_feature_matrix_not_sign= pd.DataFrame(updated_feature_matrix_not_sign)

    updated_feature_matrix_sign['class']=1
    updated_feature_matrix_not_sign['class']=0

    appended_data = pd.concat([updated_feature_matrix_sign,updated_feature_matrix_not_sign])

    # train_data = appended_data.loc[:, appended_data.columns != 'class']
    # test_data = appended_data.loc[:,appended_data.columns == 'class']


    
    kf = KFold(5, True, 2)

    for train, test in kf.split(appended_data):
        tr_data = appended_data.iloc[train]
        test_data = appended_data.iloc[test]
        x_train = tr_data.loc[:, tr_data.columns != 'class']
        y_train = tr_data['class']
        x_test = test_data.loc[:, tr_data.columns != 'class']
        y_test = test_data['class']    
        Linear_Discriminant_Analysis(x_train,x_test,y_train,y_test)
        #Quadratic_Discriminant_Analysis(x_train,x_test,y_train,y_test)
        #Logistic_Regression(x_train,x_test,y_train,y_test)
        #SupportVectorMachine(x_train, x_test, y_train, y_test)
        print('\n\n')
'''
def get_features_with_class_labels(sign,sign_num):
    all_files = glob.glob("data/"+sign+"/*.csv")
    feature_matrix = pd.DataFrame()
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_req_columns = reqcolumns(df)
        df_norm = universal_normalize(df_req_columns)
        df_features = get_features(df_norm[['leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']]) 
        feature_matrix = pd.concat([feature_matrix,df_features],ignore_index=True)
    
    feature_matrix['class'] = sign_num
    return feature_matrix

def main():
    signs = ["buy","mother","communicate","really","hope","fun"]
    sign_num = 1
    feature_matrix = pd.DataFrame()
    
    for i in signs:
        feature_matrix = get_features_with_class_labels(i, sign_num)
        sign_num = sign_num+1
        fm_after_normalization = PCA_fit(feature_matrix.loc[:,feature_matrix.columns!='class'],5,i)
        updated_feature_matrix = DimensionalityReduction(fm_after_normalization,i+"_PCA.pickle")
    
        r, c = updated_feature_matrix.shape
        updated_feature_matrix_not_sign =  updated_feature_matrix - np.random.rand(r,c)
        
        updated_feature_matrix = pd.DataFrame(updated_feature_matrix)
        updated_feature_matrix['class'] = feature_matrix['class'] 
        
        updated_feature_matrix_not_sign = pd.DataFrame(updated_feature_matrix_not_sign).loc[:,:]
        updated_feature_matrix_not_sign['class'] = 0
        
        appended_data = pd.concat([updated_feature_matrix,updated_feature_matrix_not_sign])
    #    appended_data = updated_feature_matrix
    #    print(appended_data)
        
            
        kf = KFold(5, True, 2)
    
        for train, test in kf.split(appended_data):
            tr_data = appended_data.iloc[train]
            test_data = appended_data.iloc[test]
            x_train = tr_data.loc[:, tr_data.columns != 'class']
            y_train = tr_data['class']
            x_test = test_data.loc[:, tr_data.columns != 'class']
            y_test = test_data['class']    
            Linear_Discriminant_Analysis(x_train,x_test,y_train,y_test,i)
#            Quadratic_Discriminant_Analysis(x_train,x_test,y_train,y_test,i)
#            Logistic_Regression(x_train,x_test,y_train,y_test,i)
#            SupportVectorMachine(x_train, x_test, y_train, y_test,i)
#            GradientBoostClassifier(x_train,x_test,y_train,y_test,i)
#            KNN(x_train,x_test,y_train,y_test,i)
#            Decision_tree(x_train,x_test,y_train,y_test,i)
#            GradientBoostClassifier(x_train,x_test,y_train,y_test,i)
#            AdaBoost(x_train,x_test,y_train,y_test,i)
#            RandomForest(x_train,x_test,y_train,y_test,i)
#            NaiveBayes_Classifier(x_train,x_test,y_train,y_test,i)
    
    print('\n\n')
    
if __name__ == "__main__":
    main()









    
