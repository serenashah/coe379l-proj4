from flask import Flask, request
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import joblib

app = Flask(__name__)
# load models
xgbm = xgb.Booster()
xgbm.load_model('models/xgb_model.model')
knn = joblib.load('models/knn_model.pkl')
dt = joblib.load('models/decision_tree_model.pkl')
nb = joblib.load('models/naive_bayes_model.pkl')

@app.route('/models', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "models",
      "description": "Predict flood levels using information about location, elevation, and Standard Engineering Storm Events",
   }

def preprocess_input(input):
   """
   Converts user-provided input into an dataframe that can be used with the model.
   This function could raise an exception.
   """
   # convert to a pandas df
   d = pd.DataFrame(input)
   return d

def run_model(model, d):
   if model==xgbm:
      # takes probabilities
      d = xgb.DMatrix(data=d)
      y_prob = model.predict(d)
      y_pred = np.argmax(y_prob, axis=1)
   else:
      y_pred = model.predict(d) 
   # Define mapping of numeric categories to labels
   category_labels = {0: 'Low Flood Level', 
                      1: 'Medium Flood Level', 
                      2: 'High Flood Level'}

   # Replace numeric categories with labels
   y_pred_output = np.array([category_labels[cat] for cat in y_pred])
   return y_pred_output

@app.route('/models/knn', methods=['POST'])
def classify_flood_knn():
   input = request.json.get('data')
   if not input:
      return {"error": "Please input array"}, 404
   try:
      d = preprocess_input(input)
      y_pred_knn = run_model(knn, d)
   except Exception as e:
      return {"error": f"Could not process data; details: {e}"}, 404
   return { "KNN Prediction": y_pred_knn.tolist()}

@app.route('/models/dt', methods=['POST'])
def classify_flood_dt():
   input = request.json.get('data')
   if not input:
      return {"error": "Please input array"}, 404
   try:
      d = preprocess_input(input)
      y_pred_output = run_model(dt, d)
   except Exception as e:
      return {"error": f"Could not process data; details: {e}"}, 404
   return { "Decision Tree Prediction": y_pred_output.tolist()}

@app.route('/models/nb', methods=['POST'])
def classify_flood_nb():
   input = request.json.get('data')
   if not input:
      return {"error": "Please input array"}, 404
   try:
      d = preprocess_input(input)
      y_pred_output = run_model(nb, d)
   except Exception as e:
      return {"error": f"Could not process data; details: {e}"}, 404
   return {"Naive Bayes Prediction": y_pred_output.tolist()}

@app.route('/models/xgb', methods=['POST'])
def classify_flood_xgb():
   input = request.json.get('data')
   if not input:
      return {"error": "Please input array"}, 404
   try:
      d = preprocess_input(input)
      y_pred_output = run_model(xgbm, d)
   except Exception as e:
      return {"error": f"Could not process data; details: {e}"}, 404
   return { "XGBoost Prediction": y_pred_output.tolist()}



# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')