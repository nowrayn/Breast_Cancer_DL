from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        try:
            input_data = []

            for feature in feature_names:
                value = float(request.form[feature])
                input_data.append(value)

            input_array = np.asarray(input_data).reshape(1, -1)

            # Standardize
            input_std = scaler.transform(input_array)

            # Predict
            prediction = model.predict(input_std)

            prediction_label = np.argmax(prediction)

            if prediction_label == 0:
                prediction_text = "The tumor is Malignant"
            else:
                prediction_text = "The tumor is Benign"

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction_text
    )

if __name__ == "__main__":
    app.run(debug=True)