from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        # Convert values to float where possible
        for key in input_data:
            try:
                input_data[key] = float(input_data[key])
            except:
                pass

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encoding alignment
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # Prediction
        prediction = model.predict(df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ₹ {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)