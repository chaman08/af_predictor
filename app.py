from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open('linear_regression_model (3).pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read inputs from form
        features = [
            float(request.form['l (pull)']),
            float(request.form['Cf']),
            float(request.form['P']),
            float(request.form['d']),
            float(request.form['R']),
            float(request.form['T']),
            float(request.form['Q']),
            float(request.form['UCS']),
        ]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Advancement Factor: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
