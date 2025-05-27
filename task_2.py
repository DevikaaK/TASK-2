import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  

df = pd.read_csv('titanic.csv')  

print("Summary Statistics:")
print(df.describe())

numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()

plt.figure(figsize=(15, 5 * len(numeric_cols)))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df[numeric_cols])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()
if len(numeric_cols) >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
    plt.title(f'Scatterplot between {numeric_cols[0]} and {numeric_cols[1]}')
    plt.show()

if len(numeric_cols) >= 2:
    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title='Interactive Scatterplot')
    fig.show()
