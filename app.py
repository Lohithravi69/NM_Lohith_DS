from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import joblib
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from functools import lru_cache
import random
from models import db, Merchant, Transaction, FraudLog
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder='Html')
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production
app.config['TEMPLATES_FOLDER'] = 'Html'  # Set custom templates folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Paths to the updated model, preprocessor, and dataset
model_path = 'model.pkl'
preprocessor_path = 'preprocessor.pkl'
dataset_path = 'credit_card_fraud_dataset.csv'

# Load the updated model and preprocessor
model = joblib.load(model_path)
preprocessor, feature_names = joblib.load(preprocessor_path)

# Load the dataset
df = pd.read_csv(dataset_path)

# merchant credentials are stored in the database (models.Merchant)

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

            if not merchant_id or not merchant_id.strip():
                raise ValueError("MerchantID is required.")
            if not password:
                raise ValueError("Password is required.")

            # merchant_id is now a string, no conversion needed

            # Validate MerchantID and Password using database
            merchant = Merchant.query.filter_by(merchant_id=merchant_id).first()
            if not merchant:
                flash("Invalid MerchantID or Password.", "error")
                return render_template('index.html', error="Invalid MerchantID or Password.")

            # verify hashed password
            if not check_password_hash(merchant.password, password):
                flash("Invalid MerchantID or Password.", "error")
                return render_template('index.html', error="Invalid MerchantID or Password.")

            # Store merchant_id in session
            session['merchant_id'] = merchant_id

            return redirect(url_for('dashboard'))
        except ValueError as ve:
            logging.error(f"Input Error: {ve}")
            flash(str(ve), "error")
            return render_template('index.html', error=str(ve))
        except Exception as e:
            logging.error(f"Exception Occurred: {e}")
            flash("An error occurred. Please try again.", "error")
            return render_template('index.html', error="An error occurred. Please try again.")
    return render_template('index.html', feature_names=all_columns, recent_predictions=recent_predictions)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    Route for merchant signup.
    """
    if request.method == 'POST':
        try:
            merchant_id = request.form.get('MerchantID')
            password = request.form.get('Password')
            confirm_password = request.form.get('ConfirmPassword')

            if not merchant_id or not merchant_id.strip():
                raise ValueError("MerchantID is required.")
            if not password:
                raise ValueError("Password is required.")
            if password != confirm_password:
                raise ValueError("Passwords do not match.")

            # merchant_id is now a string (allow alphanumeric IDs)
            # Check if MerchantID already exists
            existing_merchant = Merchant.query.filter_by(merchant_id=merchant_id).first()
            if existing_merchant:
                flash("MerchantID already exists. Please choose a different one.", "error")
                return render_template('signup.html', error="MerchantID already exists.")

            # Create new merchant (store hashed password)
            pw_hash = generate_password_hash(password)
            new_merchant = Merchant(merchant_id=merchant_id, password=pw_hash)
            db.session.add(new_merchant)
            db.session.commit()

            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('index'))
        except ValueError as ve:
            logging.error(f"Input Error: {ve}")
            flash(str(ve), "error")
            return render_template('signup.html', error=str(ve))
        except Exception as e:
            logging.error(f"Exception Occurred: {e}")
            flash("An error occurred. Please try again.", "error")
            return render_template('signup.html', error="An error occurred. Please try again.")
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    """
    Dashboard route showing account details and overview for logged-in merchant.
    """
    if 'merchant_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('index'))

    merchant_id = session['merchant_id']

    # Get merchant details
    merchant = Merchant.query.filter_by(merchant_id=merchant_id).first()

    # Get transaction statistics
    total_transactions = Transaction.query.filter_by(merchant_id=merchant_id).count()
    fraud_transactions = Transaction.query.filter_by(merchant_id=merchant_id).all()
    fraud_count = 0
    for tx in fraud_transactions:
        tx_data = {
            'TransactionID': tx.transaction_id,
            'TransactionDate': tx.transaction_date,
            'Amount': tx.amount,
            'MerchantID': tx.merchant_id,
            'TransactionType': tx.transaction_type,
            'Location': tx.location,
            'IsFraud': tx.is_fraud
        }
        prediction = model.predict(preprocessor.transform(pd.DataFrame([tx_data])))[0]
        if prediction == 1:
            fraud_count += 1

    # Calculate fraud ratio
    fraud_ratio = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

    return render_template('dashboard.html',
                         merchant_id=merchant_id,
                         merchant=merchant,
                         total_transactions=total_transactions,
                         fraud_count=fraud_count,
                         fraud_ratio=round(fraud_ratio, 2))

@app.route('/transactions')
def transactions():
    """
    Route to display transactions for a specific MerchantID.
    """
    if 'merchant_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('index'))

    merchant_id = session['merchant_id']

    # Query transactions from database
    transactions = Transaction.query.filter_by(merchant_id=merchant_id).all()
    transactions_data = []
    for tx in transactions:
        # Predict fraud status
        tx_data = {
            'TransactionID': tx.transaction_id,
            'TransactionDate': tx.transaction_date,
            'Amount': tx.amount,
            'MerchantID': tx.merchant_id,
            'TransactionType': tx.transaction_type,
            'Location': tx.location,
            'IsFraud': tx.is_fraud
        }
        prediction = model.predict(preprocessor.transform(pd.DataFrame([tx_data])))[0]
        tx_data['FraudStatus'] = 'Fraud' if prediction == 1 else 'No Fraud'
        transactions_data.append(tx_data)

    return render_template('transactions.html', merchant_id=merchant_id, transactions=transactions_data)

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
    if 'merchant_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('index'))

    transaction_id = request.args.get('transaction_id')
    if not transaction_id:
        return "Transaction ID required", 400

    # Query transaction from database
    transaction = Transaction.query.filter_by(transaction_id=int(transaction_id), merchant_id=session['merchant_id']).first()

    if not transaction:
        return "Transaction not found", 404

    # Add fraud status
    tx_data = {
        'TransactionID': transaction.transaction_id,
        'TransactionDate': transaction.transaction_date,
        'Amount': transaction.amount,
        'MerchantID': transaction.merchant_id,
        'TransactionType': transaction.transaction_type,
        'Location': transaction.location,
        'IsFraud': transaction.is_fraud
    }
    prediction = model.predict(preprocessor.transform(pd.DataFrame([tx_data])))[0]
    tx_data['FraudStatus'] = 'Fraud' if prediction == 1 else 'No Fraud'

    return render_template('transaction_details.html', transaction=tx_data)

@app.route('/fraud_transactions')
def fraud_transactions():
    """
    Route to display only fraud transactions (Transaction ID and Date).
    """
    if 'merchant_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('index'))

    merchant_id = session['merchant_id']

    # Query transactions from database and filter fraud ones
    transactions = Transaction.query.filter_by(merchant_id=merchant_id).all()
    fraud_transactions_data = []
    for tx in transactions:
        tx_data = {
            'TransactionID': tx.transaction_id,
            'TransactionDate': tx.transaction_date,
            'Amount': tx.amount,
            'MerchantID': tx.merchant_id,
            'TransactionType': tx.transaction_type,
            'Location': tx.location,
            'IsFraud': tx.is_fraud
        }
        prediction = model.predict(preprocessor.transform(pd.DataFrame([tx_data])))[0]
        if prediction == 1:
            tx_data['FraudStatus'] = 'Fraud'
            fraud_transactions_data.append(tx_data)

    return render_template(
        'fraud_transactions.html',
        merchant_id=merchant_id,
        fraud_transactions=fraud_transactions_data
    )

@app.route('/logout')
def logout():
    """
    Route to log out the user.
    """
    session.pop('merchant_id', None)
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Flask app. Access it at: http://127.0.0.1:5000")
    app.run(debug=True)
