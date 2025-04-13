
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the encoded dataset
df = pd.read_csv("Encoded_Customer_support_data.csv")

# Keep only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Save correlation matrix to CSV
correlation_matrix.to_csv("Correlation_Matrix.csv")

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()

# Save the heatmap
plt.savefig("correlation_heatmap.png")
print("Correlation analysis complete. Matrix and heatmap saved.")
