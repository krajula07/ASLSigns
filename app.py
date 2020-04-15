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

get_list = {1: 'buy', 2: 'mother', 3: 'communicate',
            4: 'really', 5: 'hope', 6: 'fun'}

random_forest_model = pickle.load(open('random_forest.pkl', 'rb'))
lda_model = pickle.load(open('lda.pkl', 'rb'))
svm_model = pickle.load(open('svm.pkl', 'rb'))
gradient_boost_model = pickle.load(open('gradientboost.pkl', 'rb'))


def json_to_dataframe(test_json):
    all_frames = []
    for i in range(len(test_json)):
        temp = []
        current_frame = test_json[i]
        temp.append(current_frame['score'])
        key_points = current_frame['keypoints']
        for j in range(len(key_points)):
            parts = key_points[j]
            temp.append(parts['score'])
            position = parts['position']
            x, y = list(position.values())
            temp.append(x)
            temp.append(y)
        all_frames.append(temp)
    data_frame = pandas.DataFrame(all_frames, columns=columns)
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

    output = {'1': get_list[random_forest_model_predictions[0]], '2': get_list[lda_model_predictions[0]],
              '3': get_list[svm_model_predictions[0]], '4': get_list[gradient_boost_model_predictions[0]]}
    return output


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
