import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the updated dataset
dataset_path = r'c:\Users\lohit\Documents\gproject\flask-app\credit_card_fraud_dataset_updated.csv'
df = pd.read_csv(dataset_path)

# Preprocess the data
X = df[['Amount', 'TransactionType', 'Location']]
y = df['IsFraud']

categorical_features = ['TransactionType', 'Location']
numerical_features = ['Amount']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model and preprocessor
model_path = r'c:\Users\lohit\Documents\gproject\flask-app\model.pkl'
preprocessor_path = r'c:\Users\lohit\Documents\gproject\flask-app\preprocessor.pkl'
joblib.dump(pipeline.named_steps['classifier'], model_path)
joblib.dump((preprocessor, X.columns.tolist()), preprocessor_path)

print("Model and preprocessor saved successfully!")