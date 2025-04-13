
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Customer_support_data.csv")

# --- Convert datetime for response duration calculation ---
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce', dayfirst=True)
df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce', dayfirst=True)
df['Response_Duration'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 3600  # Convert to hours

# --- Binning Response_Duration (in hours) ---
duration_bins = [0, 2, 8, 24, 72, np.inf]
duration_labels = ['Immediate (0-2h)', 'Quick (2-8h)', 'Moderate (8-24h)', 'Slightly Delayed (24-72h)','Delayed (>3d)']
df['Response_Duration_Bin'] = pd.cut(df['Response_Duration'], bins=duration_bins, labels=duration_labels, include_lowest=True)

# --- Binning Item Price (if present) ---
if 'Item Price' in df.columns:
    price_bins = [0, 500, 2000, np.inf]
    price_labels = ['Low (<500)', 'Mid (500-2000)', 'High (>2000)']
    df['Item_Price_Bin'] = pd.cut(df['Item Price'], bins=price_bins, labels=price_labels, include_lowest=True)

# --- Binning CSAT Score (if numeric) ---
if pd.api.types.is_numeric_dtype(df['CSAT Score']):
    csat_bins = [0, 2, 3, 5]
    csat_labels = ['Low (1-2)', 'Medium (3)', 'High (4-5)']
    df['CSAT_Bin'] = pd.cut(df['CSAT Score'], bins=csat_bins, labels=csat_labels, include_lowest=True)

# Save binned data for inspection
df.to_csv("Binned_Customer_support_data.csv", index=False)
print("Binned columns (duration in hours) created and saved to 'Binned_Customer_support_data.csv'")
