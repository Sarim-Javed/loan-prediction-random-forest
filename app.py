from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the Random Forest model
random_forest_model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = {
        'Income': float(request.form['Income']),
        'Age': int(request.form['Age']),
        'Experience': int(request.form['Experience']),
        'Married/Single': request.form['Married/Single'],
        'House_Ownership': request.form['House_Ownership'],
        'Car_Ownership': request.form['Car_Ownership'],
        'Profession': request.form['Profession'],
        'CITY': request.form['CITY'],
        'STATE': request.form['STATE'],
        'CURRENT_JOB_YRS': int(request.form['CURRENT_JOB_YRS']),
        'CURRENT_HOUSE_YRS': int(request.form['CURRENT_HOUSE_YRS'])
    }

    # Convert input data to DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = random_forest_model.predict(input_data_df)

    # Convert prediction to human-readable format
    result = 'Low Risk' if prediction[0] == 0 else 'High Risk'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
