from flask import Flask, render_template_string, request
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("fraud_detection_model.pkl")

app = Flask(__name__)

# HTML template with professional styling
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>UPI TICK</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #fff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            width: 400px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        button {
            margin-top: 20px;
            width: 100%;
            padding: 12px;
            background: #6a11cb;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: #fff;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        button:hover {
            opacity: 0.9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            font-weight: bold;
        }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fraud Detection App</h1>
        <form method="POST">
            <label>Transaction Type</label>
            <select name="transaction_type">
                <option value="PAYMENT">PAYMENT</option>
                <option value="TRANSFER">TRANSFER</option>
                <option value="WITHDRAWAL">WITHDRAWAL</option>
                <option value="DEPOSIT">DEPOSIT</option>
            </select>
            <label>Transaction Amount</label>
            <input type="number" name="amount" min="0" step="0.01" value="1000">
            <label>Old Balance (Sender)</label>
            <input type="number" name="oldbalanceOrg" min="0" step="0.01" value="10000">
            <label>New Balance (Sender)</label>
            <input type="number" name="newbalanceOrig" min="0" step="0.01" value="9000">
            <label>Old Balance (Receiver)</label>
            <input type="number" name="oldbalanceDest" min="0" step="0.01" value="0">
            <label>New Balance (Receiver)</label>
            <input type="number" name="newbalanceDest" min="0" step="0.01" value="0">
            <button type="submit">Predict</button>
        </form>

        {% if prediction is not none %}
            <div class="result {% if prediction == 1 %}error{% else %}success{% endif %}">
                Prediction: {{ prediction }} - 
                {% if prediction == 1 %} This Transaction may be FRAUD {% else %} This Transaction is SAFE {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Read inputs and convert types properly
            transaction_type = str(request.form.get("transaction_type"))
            amount = float(request.form.get("amount", 0))
            oldbalanceOrg = float(request.form.get("oldbalanceOrg", 0))
            newbalanceOrig = float(request.form.get("newbalanceOrig", 0))
            oldbalanceDest = float(request.form.get("oldbalanceDest", 0))
            newbalanceDest = float(request.form.get("newbalanceDest", 0))

            # Create DataFrame with exact types
            input_data = pd.DataFrame({
                "type": [transaction_type],
                "amount": [amount],
                "oldbalanceOrg": [oldbalanceOrg],
                "newbalanceOrig": [newbalanceOrig],
                "oldbalanceDest": [oldbalanceDest],
                "newbalanceDest": [newbalanceDest]
            })

            # Prediction
            prediction = int(model.predict(input_data)[0])

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(html_template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
