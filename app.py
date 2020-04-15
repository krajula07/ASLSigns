from flask import Flask
from flask import json
from flask import request
import pandas
import pickle
from train_combined_features import *

app = Flask(__name__)

columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
           'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
           'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
           'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
           'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
           'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
           'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
           'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
           'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

signnum_sign_dict = {1: 'buy', 2: 'mother', 3: 'communicate',
                     4: 'really', 5: 'hope', 6: 'fun'}

random_forest_model = pickle.load(open('random_forest.pkl', 'rb'))
lda_model = pickle.load(open('lda.pkl', 'rb'))
svm_model = pickle.load(open('svm.pkl', 'rb'))
gradient_boost_model = pickle.load(open('gradientboost.pkl', 'rb'))


def json_to_dataframe(test_json):
    data = request.json
    dataframe_data = np.zeros((len(data), len(columns)))
    for i in range(dataframe_data.shape[0]):
        row = []
        row.append(data[i]['score'])
        for each_row in data[i]['keypoints']:
            row.append(each_row['score'])
            row.append(each_row['position']['x'])
            row.append(each_row['position']['y'])
        dataframe_data[i] = np.array(row)
    data_frame = pd.DataFrame(dataframe_data, columns=columns)
    return data_frame


@app.route('/', methods=['POST'])
def test_predict():
    test_json = request.json
    test_data_frame = json_to_dataframe(test_json)
    features = features_get(test_data_frame)
    random_forest_model_predictions = random_forest_model.predict(features)
    lda_model_predictions = lda_model.predict(features)
    svm_model_predictions = svm_model.predict(features)
    gradient_boost_model_predictions = gradient_boost_model.predict(features)

    json_output = {'1': signnum_sign_dict.get(random_forest_model_predictions[0]), '2': signnum_sign_dict.get(lda_model_predictions[0]),
                   '3': signnum_sign_dict.get(svm_model_predictions[0]), '4': signnum_sign_dict.get(gradient_boost_model_predictions[0])}
    return json_output


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
