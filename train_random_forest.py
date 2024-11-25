import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = "city_day (1).csv"  # Replace with your CSV path
data = pd.read_csv(file_path)

# Drop rows where the target variable is missing
data_cleaned = data.dropna(subset=['AQI_Bucket'])

# Fill missing values in numeric columns with the mean
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data_cleaned[numeric_columns] = imputer.fit_transform(data_cleaned[numeric_columns])

# Encode the target variable
label_encoder = LabelEncoder()
data_cleaned['AQI_Bucket'] = label_encoder.fit_transform(data_cleaned['AQI_Bucket'])

# Exclude 'AQI' and any other non-relevant features from the feature set
data_cleaned = data_cleaned.drop(columns=['City', 'Date', 'AQI'])

# Define features and target (only the first 12 features)
X = data_cleaned.iloc[:, :12]  # Use the first 12 features
y = data_cleaned['AQI_Bucket']

# Save feature names for reference during prediction
feature_names = list(X.columns)
joblib.dump(feature_names, "feature_names.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
