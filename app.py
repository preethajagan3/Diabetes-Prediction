import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

# Load the trained model 
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    

app = Flask(__name__,template_folder='template')

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()  # Get data from form

        # Convert string values to numbers
        features = [float(data.get(key, 0)) for key in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 10, 2) 
        return render_template('result.html', prediction=prediction)

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    print("We deployed successfully!!")
