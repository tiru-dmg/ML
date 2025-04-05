import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: better theme for plots
sns.set(style="whitegrid")

# 🔹 Small Sample Dataset
data = {
    'Age': [22, 35, 26, 45, 33],
    'Income': [25000, 48000, 32000, 54000, 40000],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Purchased': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 1️⃣ Histogram with KDE for Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=5, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2️⃣ Histogram with KDE for Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Income'], bins=5, kde=True, color='salmon')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# ✅ Count Plot for Gender Distribution (Fixed FutureWarning)
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Gender', data=df, palette='Set2', legend=False)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# ✅ Count Plot for Purchased Class Distribution (Fixed FutureWarning)
plt.figure(figsize=(6,4))
sns.countplot(x='Purchased', hue='Purchased', data=df, palette='Set1', legend=False)
plt.title('Class Distribution of Purchase')
plt.xlabel('Purchased (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# 5️⃣ Correlation Heatmap (Heatmap of Numeric Correlation)
plt.figure(figsize=(6,5))
correlation = df[['Age', 'Income', 'Purchased']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 6️⃣ Pair Plot (Scatter plots + Histograms to show relation between variables)
sns.pairplot(df, hue='Purchased', palette='husl')
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()
