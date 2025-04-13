
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

# Load dataset
df = pd.read_csv("Binned_Customer_support_data.csv")

# ------- Ordinal Encoding --------
# Tenure Bucket (example order)
tenure_order = ['< 3 months', '3-6 months', '6-12 months', '1-2 years', '> 2 years']
if 'Tenure Bucket' in df.columns:
    df['Tenure_Bucket_Encoded'] = pd.Categorical(df['Tenure Bucket'], categories=tenure_order, ordered=True).codes

# Agent Shift (assumed order)
shift_order = ['Morning', 'Afternoon', 'Night']
if 'Agent Shift' in df.columns:
    df['Agent_Shift_Encoded'] = pd.Categorical(df['Agent Shift'], categories=shift_order, ordered=True).codes

# ------- Label Encoding --------
label_cols = ['issue_type', 'Product_category', 'Sub-category', 'channel_name', 'Customer_City']
for col in label_cols:
    if col in df.columns:
        df[col + '_Encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

# ------- Target Encoding for High Cardinality --------
target_cols = ['Agent_name', 'Supervisor', 'Manager']
for col in target_cols:
    if col in df.columns and 'CSAT Score' in df.columns:
        target_encoder = ce.TargetEncoder()
        df[col + '_Encoded'] = target_encoder.fit_transform(df[col], df['CSAT Score'])

# Save encoded dataset
df.to_csv("Encoded_Customer_support_data.csv", index=False)
print("Encoding complete. Output saved as 'Encoded_Customer_support_data.csv'")
