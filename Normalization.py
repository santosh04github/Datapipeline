
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load encoded dataset
df = pd.read_csv("Encoded_Customer_support_data.csv")

# Select numeric columns (excluding CSAT Score if it's the target)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if 'CSAT Score' in numeric_cols:
    numeric_cols.remove('CSAT Score')

# Apply Min-Max normalization
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the normalized dataset
df.to_csv("Normalized_Customer_support_data.csv", index=False)
print("Normalization complete. Normalized data saved to 'Normalized_Customer_support_data.csv'")
