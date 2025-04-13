
import pandas as pd
from scipy.stats import chi2_contingency

# Load the encoded dataset
df = pd.read_csv("Encoded_Customer_support_data.csv")

# Define the target column
target_col = 'CSAT Score'

# Select categorical columns (object or category type)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# If there are encoded categorical features as numeric, add them manually
# Example: 'Tenure Bucket_Encoded', 'Agent_Shift_Encoded', etc.
additional_cat_cols = ['Tenure Bucket', 'Agent_Shift', 'issue_type', 'Product Name', 'Category', 'Sub Category']
for col in additional_cat_cols:
    if col in df.columns and col not in categorical_cols:
        categorical_cols.append(col)

# Perform Chi-Square test for each categorical column against CSAT Score
results = []
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df[target_col])
    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results.append({'Feature': col, 'Chi2': chi2, 'p-value': p})
    except Exception as e:
        results.append({'Feature': col, 'Chi2': None, 'p-value': None, 'Error': str(e)})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("ChiSquare_Results.csv", index=False)
print("Chi-Square test completed. Results saved to 'ChiSquare_Results.csv'")
