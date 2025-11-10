from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from functools import lru_cache
import random

app = Flask(__name__, template_folder='Html')
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production
app.config['TEMPLATES_FOLDER'] = 'Html'  # Set custom templates folder

# Paths to the updated model, preprocessor, and dataset
model_path = 'model.pkl'
preprocessor_path = 'preprocessor.pkl'
dataset_path = 'credit_card_fraud_dataset.csv'

# Load the updated model and preprocessor
model = joblib.load(model_path)
preprocessor, feature_names = joblib.load(preprocessor_path)

# Load the dataset
df = pd.read_csv(dataset_path)

# Example MerchantID and Passwords (replace with a secure database in production)
merchant_credentials = {
    999: "password123",
    888: "securepass",
    777: "merchant777"
}

# Dynamically load column names from the dataset
all_columns = df.columns.tolist()
all_columns.remove('IsFraud')  # Exclude the target column

print("=== Feature Names from Dataset ===")
print(all_columns)

# Configure logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_input(data, preprocessor):
    """
    Preprocess the input data using the preprocessor.
    """
    df = pd.DataFrame([data])
    try:
        df = df[feature_names]
    except KeyError as e:
        raise ValueError(f"Input features do not match expected features: {e}")
    transformed = preprocessor.transform(df)
    return transformed

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for the homepage.
    """
    recent_predictions = []  # Replace with logic to fetch recent predictions
    if request.method == 'POST':
        try:
            # Get MerchantID and Password from the form
            merchant_id = request.form.get('MerchantID')
            password = request.form.get('Password')

            if not merchant_id or not merchant_id.strip().isdigit():
                raise ValueError("MerchantID is required and must be a number.")
            if not password:
                raise ValueError("Password is required.")

            merchant_id = int(merchant_id)

            # Validate MerchantID and Password
            if (merchant_id not in merchant_credentials or 
                merchant_credentials[merchant_id] != password):
                return render_template('index.html', error="Invalid MerchantID or Password.")

            # Filter transactions for the given MerchantID
            merchant_transactions = df[df['MerchantID'] == merchant_id]

            if merchant_transactions.empty:
                return redirect(url_for('transactions', merchant_id=merchant_id, error="No transactions found for this MerchantID."))

            # Add fraud predictions to the transactions
            X = merchant_transactions.drop(['IsFraud'], axis=1)
            predictions = model.predict(preprocessor.transform(X))
            merchant_transactions['FraudStatus'] = ['Fraud' if pred == 1 else 'No Fraud' for pred in predictions]

            # Save the filtered transactions to display on the next page
            merchant_transactions.to_csv('filtered_transactions.csv', index=False)

            return redirect(url_for('transactions', merchant_id=merchant_id))
        except ValueError as ve:
            logging.error(f"Input Error: {ve}")
            return render_template('index.html', error=str(ve))
        except Exception as e:
            logging.error(f"Exception Occurred: {e}")
            return render_template('index.html', error="An error occurred. Please try again.")
    return render_template('index.html', feature_names=all_columns, recent_predictions=recent_predictions)

@app.route('/transactions')
def transactions():
    """
    Route to display transactions for a specific MerchantID.
    """
    merchant_id = request.args.get('merchant_id', 'Unknown')
    error = request.args.get('error', None)

    if error:
        return render_template('transactions.html', merchant_id=merchant_id, error=error, transactions=None)

    # Load the filtered transactions
    merchant_transactions = pd.read_csv('filtered_transactions.csv')
    return render_template('transactions.html', merchant_id=merchant_id, transactions=merchant_transactions.to_dict(orient='records'))

@app.route('/result')
def result():
    """
    Route to display the prediction result.
    """
    prediction = request.args.get('prediction', 'No result available')
    return render_template('result.html', prediction=prediction)

@lru_cache(maxsize=None)
def get_fraud_ratio():
    fraud_count = df['IsFraud'].sum()
    total_count = len(df)
    return fraud_count / total_count if total_count > 0 else 0

@lru_cache(maxsize=None)
def get_feature_importance():
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance and take top 10
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        return dict(sorted_importance)
    else:
        # Fallback if no feature importances
        return {col: 1.0 / len(feature_names) for col in feature_names[:10]}

@app.route('/insights')
def insights():
    """
    Route to display model insights.
    """
    fraud_ratio = get_fraud_ratio()
    feature_importance = get_feature_importance()
    return render_template('insights.html', fraud_ratio=fraud_ratio, feature_importance=feature_importance)

@app.route('/api/live_feed')
def api_live_feed():
    """
    API endpoint for live fraud detection feed.
    """
    # Simulate live data by randomly selecting transactions
    sample_size = min(10, len(df))
    sample_transactions = df.sample(n=sample_size, random_state=random.randint(0, 1000))
    
    live_feed = []
    for _, row in sample_transactions.iterrows():
        transaction_data = row.drop('IsFraud').to_dict()
        prediction = model.predict(preprocessor.transform(pd.DataFrame([transaction_data])))[0]
        result = 'Fraud Detected' if prediction == 1 else 'No Fraud Detected'
        live_feed.append({
            'TransactionID': int(row['TransactionID']),
            'Amount': float(row['Amount']),
            'Result': result
        })
    
    return jsonify(live_feed)

@app.route('/live')
def live_detection():
    """
    Route to display live detection feed.
    """
    live_feed = []  # Initial empty feed, will be populated via AJAX
    return render_template('live.html', live_feed=live_feed)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    """
    Admin portal for uploading datasets and retraining the model.
    """
    if request.method == 'POST':
        file = request.files['dataset']
        file.save('uploaded_dataset.csv')  # Save the uploaded dataset
        df = pd.read_csv('uploaded_dataset.csv')
        X = df.drop('IsFraud', axis=1)
        y = df['IsFraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        new_model = RandomForestClassifier(random_state=42)
        new_model.fit(X_train, y_train)
        joblib.dump(new_model, model_path)
        return "Dataset uploaded and model retrained successfully!"
    return render_template('admin.html')

@app.route('/transaction_details')
def transaction_details():
    """
    Route to display full details of a specific transaction.
    """
    transaction_id = request.args.get('transaction_id')

    # Load the filtered transactions
    merchant_transactions = pd.read_csv('filtered_transactions.csv')

    # Get the specific transaction details
    transaction = merchant_transactions[merchant_transactions['TransactionID'] == int(transaction_id)].to_dict(orient='records')

    if not transaction:
        return "Transaction not found", 404

    return render_template('transaction_details.html', transaction=transaction[0])

@app.route('/fraud_transactions')
def fraud_transactions():
    """
    Route to display only fraud transactions (Transaction ID and Date).
    """
    merchant_id = request.args.get('merchant_id', 'Unknown')

    # Load the filtered transactions
    merchant_transactions = pd.read_csv('filtered_transactions.csv')

    # Filter only fraud transactions
    fraud_transactions = merchant_transactions[merchant_transactions['FraudStatus'] == 'Fraud']

    return render_template(
        'fraud_transactions.html',
        merchant_id=merchant_id,
        fraud_transactions=fraud_transactions.to_dict(orient='records')
    )

if __name__ == '__main__':
    print("Starting Flask app. Access it at: http://127.0.0.1:5000")
    app.run(debug=True)