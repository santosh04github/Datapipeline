
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load encoded dataset
df = pd.read_csv("Encoded_Customer_support_data.csv")

# Set plot style
sns.set(style="whitegrid")

# Plot 1: CSAT Score distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='CSAT Score', hue='CSAT Score', palette='viridis', legend=False)
plt.title("Distribution of CSAT Scores")
plt.xlabel("CSAT Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("csat_score_distribution.png")
plt.close()

# Plot 2: Boxplot - Item Price vs CSAT Score (if exists)
if 'Item Price' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='CSAT Score', y='Item Price', hue='CSAT Score', palette='coolwarm', legend=False)
    plt.title("Item Price vs CSAT Score")
    plt.xlabel("CSAT Score")
    plt.ylabel("Item Price")
    plt.tight_layout()
    plt.savefig("item_price_vs_csat.png")
    plt.close()

# Plot 3: Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Plot 4: Response Duration vs CSAT Score (if exists)
if 'Response_Duration' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='CSAT Score', y='Response_Duration', hue='CSAT Score', palette='magma', legend=False)
    plt.title("Response Duration vs CSAT Score")
    plt.xlabel("CSAT Score")
    plt.ylabel("Response Duration (in hours)")
    plt.tight_layout()
    plt.savefig("response_duration_vs_csat.png")
    plt.close()

print("Visualizations created and saved.")
