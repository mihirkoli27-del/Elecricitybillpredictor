from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model

model = pickle.load(open("model.pkl", "rb"))

# Season mapping

season_map = {
    "Summer": 1,
    "Winter": 2,
    "Monsoon": 0
}

@app.route('/')

def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])

def predict():
    units = float(request.form['units'])
    appliances = int(request.form['appliances'])
    hours = float(request.form['hours'])
    season = request.form['season']
    rate = float(request.form['rate'])

    season_encoded = season_map.get(season, 0)

    features = np.array([[units, appliances, hours, season_encoded, rate]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Bill: ₹{round(prediction[0], 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)
