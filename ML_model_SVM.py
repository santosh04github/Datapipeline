import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('Customer_support_data.csv').head(10000)

# Convert CSAT Score to binary target: 1 = satisfied (score ≥ 4), 0 = not satisfied (score < 4)
df['satisfied'] = df['CSAT Score'].apply(lambda x: 1 if x >= 4 else 0)

# Drop irrelevant or high-null columns
drop_cols = ['Unique id', 'Customer Remarks', 'Order_id', 'order_date_time',
             'Issue_reported at', 'issue_responded', 'Survey_response_Date',
             'Customer_City', 'Product_category', 'connected_handling_time']
df.drop(columns=drop_cols, inplace=True)

# Handle missing values by filling with mean for numerical columns and mode for categorical ones
df.fillna(df.mean(), inplace=True)  # Filling numerical NaN values with column mean
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Filling categorical NaN with mode

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop('satisfied', axis=1)
y = df_encoded['satisfied']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
print("Starting model training")
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
print(" Model training complete.")


# Predict
y_pred = svm_model.predict(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create report
report = f"""
Sub-Objective 2: Design and Development of a Machine Learning Pipeline

2.1 Model Preparation:
- Selected Algorithm: Support Vector Machine (SVM)
- Reason: Effective in high-dimensional space and well-suited for classification.

2.2 Model Training:
- 70% training and 30% testing
- Categorical variables encoded using one-hot encoding
- Feature scaling applied with StandardScaler
- Missing values handled with mean (numerical) and mode (categorical)

2.3 Model Evaluation:
- Accuracy: {accuracy:.4f}

2.4 MLOps: Metrics Logged
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1 Score: {f1:.4f}
"""

# Save report to file
with open("svm_pipeline_report.txt", "w") as f:
    f.write(report)

print("✅ Report saved to 'svm_pipeline_report.txt'")
