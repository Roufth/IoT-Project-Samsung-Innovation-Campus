import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset (simpan file dataset di Working Directory yang sama)
data = pd.read_csv('ai4i2020.csv')  

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables (Product Type)
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])

# Features and target variable
features = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
target = data['TWF']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the Random Forest classifier on the resampled data
rf_model_resampled = RandomForestClassifier(random_state=42)
rf_model_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = rf_model_resampled.predict(X_test_scaled)

# Evaluate the model
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
classification_rep_resampled = classification_report(y_test, y_pred_resampled)
conf_matrix_resampled = confusion_matrix(y_test, y_pred_resampled)

# Print the results
print("Accuracy:", accuracy_resampled)
print("Classification Report:\n", classification_rep_resampled)
print("Confusion Matrix:\n", conf_matrix_resampled)
