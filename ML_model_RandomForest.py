import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('Customer_support_data.csv').head(10000)

# Create binary target: 1 = satisfied (score ≥ 4), 0 = not satisfied (score < 4)
df['satisfied'] = df['CSAT Score'].apply(lambda x: 1 if x >= 4 else 0)

# Drop target-related column to avoid leakage
df.drop(columns=['CSAT Score'], inplace=True)

# Drop irrelevant or high-null columns
drop_cols = ['Unique id', 'Customer Remarks', 'Order_id', 'order_date_time',
             'Issue_reported at', 'issue_responded', 'Survey_response_Date',
             'Customer_City', 'Product_category', 'connected_handling_time']
df.drop(columns=drop_cols, inplace=True, errors='ignore')  # in case some don't exist

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df_encoded.drop('satisfied', axis=1)
y = df_encoded['satisfied']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest Classifier
print("✅ Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Report
report = f"""
Sub-Objective 2: Design and Development of a Machine Learning Pipeline

2.1 Model Preparation:
- Selected Algorithm: Random Forest Classifier
- Reason: Handles large datasets efficiently and works well with both categorical and numerical data.

2.2 Model Training:
- 70% training and 30% testing
- Stratified sampling to handle class imbalance
- Categorical variables encoded using one-hot encoding
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
with open("random_forest_report.txt", "w") as f:
    f.write(report)

print("✅ Metrics calculated and report saved to 'random_forest_report.txt'")
