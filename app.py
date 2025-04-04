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
            float(request.form['l_pull']),
            float(request.form['cf']),
            float(request.form['p']),
            float(request.form['hole_depth']),
            float(request.form['r']),
            float(request.form['total_ch']),
            float(request.form['q']),
            float(request.form['ucs']),
        ]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Advancement Factor: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
