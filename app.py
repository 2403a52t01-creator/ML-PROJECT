from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()

    # Convert numeric form values from strings to numbers.
    for key, value in input_data.items():
        try:
            input_data[key] = float(value)
        except ValueError:
            input_data[key] = value

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return render_template("index.html", prediction_text=f"Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)