from flask import Flask, request, jsonify
from modules.insurance_model import InsuranceModel
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return "API Modelling Prediction"

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    result_predict = InsuranceModel().runModel(df, typed='single')
    return jsonify(
        {
            "status":"predicted",
            "predicted_result":result_predict           
        }
    )

if __name__ == "__main__":
    app.run()