from __future__ import print_function
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import json
import warnings
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://project-2-app.vercel.app/"])


@app.route("/")
def hello_world():
    return "Welcome to the backend"


@app.route("/predict-crop", methods=["POST"])
def recommend_crop():
    # Create a dictionary with the data you want to return as JSON
    data = request.json
    # NPK is the soil nitrogen, phosphorus and potassium
    # The input is a whole number but in reality it has to be divided by 10. example: 96 >> 9.6
    # N = data["N"]
    # P = data["P"]
    # K = data["K"]
    # ph = data["ph"]

    N = 10
    P = 20
    K = 15
    ph = 5.0
    temperature = data["temperature"]
    humidity = data["humidity"]
    rainfall = data["rainfall"]

    PATH = "./Crop_recommendation.csv"
    crop = pd.read_csv(PATH)
    features = crop[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    target = crop["label"]

    # Splitting into train and test data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        features, target, test_size=0.2, random_state=2
    )

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain, Ytrain)

    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)

    # these are the expected input [['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = RF.predict(input_data)

    # Create a dictionary to store the prediction result
    result_dict = {
        "prediction": prediction.tolist()
    }  # Convert the NumPy array to a Python list

    # Serialize the result_dict to JSON
    json_result = json.dumps(result_dict)

    # Use jsonify to convert the dictionary to a JSON response
    return json_result


@app.route("/predict-yield", methods=["POST"])
def predict_yield():
    df = pd.read_csv("./yield_df.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)

    def isStr(obj):
        try:
            float(obj)
            return False
        except:
            return True

    to_drop = df[df["average_rain_fall_mm_per_year"].apply(isStr)].index

    df = df.drop(to_drop)
    df["average_rain_fall_mm_per_year"] = df["average_rain_fall_mm_per_year"].astype(
        np.float64
    )
    crops = df["Item"].unique()
    yield_per_crop = []
    for crop in crops:
        yield_per_crop.append(
            df[df["Item"] == crop]["hg/ha_yield"].sum()
        )  # Train Test split Rearranging Columns

    col = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "Area",
        "Item",
        "hg/ha_yield",
    ]
    df = df[col]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0, shuffle=True
    )
    ohe = OneHotEncoder(drop="first")
    scale = StandardScaler()

    preprocesser = ColumnTransformer(
        transformers=[
            ("StandardScale", scale, [0, 1, 2, 3]),
            ("OHE", ohe, [4, 5]),
        ],
        remainder="passthrough",
    )
    X_train_dummy = preprocesser.fit_transform(X_train)
    X_test_dummy = preprocesser.transform(X_test)
    preprocesser.get_feature_names_out(col[:-1])

    dtr = DecisionTreeRegressor()
    dtr.fit(X_train_dummy, y_train)
    dtr.predict(X_test_dummy)

    # replace this with request data
    data = request.json
    year = 2013  # use 2013 as year
    average_rain_fall_mm_per_year = data["average_rain_fall_mm_per_year"]
    pesticides_tonnes = data["pesticides_tonnes"]
    avg_temp = data["avg_temp"]
    area = "Canada"  # default to Canada
    item = data["item"]
    print(data)

    features = np.array(
        [
            [
                year,
                average_rain_fall_mm_per_year,
                pesticides_tonnes,
                avg_temp,
                area,
                item,
            ]
        ],
        dtype=object,
    )

    # Transform the features using the preprocessor
    transformed_features = preprocesser.transform(features)

    # Make the prediction
    predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

    # Create a dictionary to store the prediction result
    result_dict = {
        "predicted_yield": predicted_yield[0][0].tolist()
    }  # Convert the NumPy array to a Python list

    # Serialize the result_dict to JSON
    json_result = json.dumps(result_dict)

    # Use jsonify to convert the dictionary to a JSON responxse
    # 1 hectogram/hectare = 0.01grams/sqm
    return json_result  # this is hectogram/hectare


if __name__ == "__main__":
    app.run(debug=True, port=5000)
