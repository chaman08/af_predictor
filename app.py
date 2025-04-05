from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("linear_regression_model (4).pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        input_features = [
            float(request.form['W']),
            float(request.form['N (no. of holes)']),
            float(request.form['T (total ch)']),
            float(request.form['q']),
            float(request.form['Cf']),
            float(request.form['UCS']),
            float(request.form['Q']),
            float(request.form[' d(hole depth)'])
        ]

        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return render_template('index.html', prediction_text=f'Predicted Advancement Factor: {prediction:.2f}')
    
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)