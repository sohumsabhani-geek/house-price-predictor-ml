import pickle
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from waitress import serve

logging.basicConfig(level=logging.INFO)

# Load model
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('House Price')
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # FIX 1: must be list
        X_data = dv.transform([data])

        # predict
        y_pred = model.predict(X_data)

        # FIX 2: take first value
        result = {
            'price': int(y_pred[0])
        }

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def form():
    return render_template("index.html")


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)