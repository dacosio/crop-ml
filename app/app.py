from __future__ import print_function
from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import warnings
from flask_cors import CORS
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")
@app.route('/predict-crop', methods=['POST'])
def recommend_crop():
    # Create a dictionary with the data you want to return as JSON
    data = request.json
    N = data["N"]
    P = data["P"]
    K = data["K"]
    temperature = data["temperature"]
    humidity = data["humidity"]
    ph = data["ph"]
    rainfall = data["rainfall"]


    PATH = './Crop_recommendation.csv'
    crop  = pd.read_csv(PATH)
    features = crop[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = crop['label']

    # Splitting into train and test data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)

    # these are the expected input [['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    input_data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction = RF.predict(input_data)

    # Create a dictionary to store the prediction result
    result_dict = {'prediction': prediction.tolist()}  # Convert the NumPy array to a Python list

    # Serialize the result_dict to JSON
    json_result = json.dumps(result_dict)
    
    
    # Use jsonify to convert the dictionary to a JSON response
    return json_result

if __name__ == '__main__':
    app.run(debug=True, port=5000)