# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('linear_regression_model (5).pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Fetch all input fields
            W = float(request.form['W'])
            N = float(request.form['N'])
            T = float(request.form['T'])
            q = float(request.form['q'])
            Cf = float(request.form['Cf'])
            UCS = float(request.form['UCS'])
            RQD = float(request.form['RQD'])
            d = float(request.form['d'])

            # Prepare the input for prediction
            input_features = np.array([[W, N, T, q, Cf, UCS, RQD, d]])

            # Make prediction
            prediction = model.predict(input_features)[0]

            return render_template('index.html', prediction=prediction)
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
