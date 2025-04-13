import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Customer_support_data.csv")
print(df)

# Convert date columns to datetime format
df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'], format='%d-%m-%Y %H:%M',errors='coerce', dayfirst=True)
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], format='%d-%m-%Y %H:%M',errors='coerce', dayfirst=True)
df['issue_responded'] = pd.to_datetime(df['issue_responded'], format='%d-%m-%Y %H:%M',errors='coerce', dayfirst=True)
# Calculate response time in days
df['Response_Duration'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 86400  # Convert seconds to days

# Display a few samples
print("\nResponse Duration (in days):")
print(df[['Issue_reported at', 'issue_responded', 'Response_Duration']].head())

# Calculate and print the average response time
average_response_time = df['Response_Duration'].mean()
print(f"\n Average Response Time: {average_response_time:.2f} days")

# Segmenting by issue type (if the column exists)
if 'issue_type' in df.columns:
    # Group by issue type and calculate the average response time
    avg_response_by_issue = df.groupby('issue_type')['Response_Duration'].mean().reset_index()

    # Print the segmented data
    print("\n Average Response Time by Issue Type:")
    print(avg_response_by_issue)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing issue_type with 'Unknown'
if 'issue_type' in df.columns:
    df['issue_type'].fillna('Unknown', inplace=True)

# Fill missing Response_Duration with median
if 'Response_Duration' in df.columns:
    df['Response_Duration'] = df['Response_Duration'].fillna(df['Response_Duration'].median())
    
# Drop rows where key dates are missing
df.dropna(subset=['Survey_response_Date', 'Issue_reported at', 'issue_responded'], inplace=True)

# Data type information
print("\nData Types:")
print(df.dtypes)