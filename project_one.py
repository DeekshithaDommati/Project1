import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Basic information about the dataset
print(iris.info())

# Summary statistics
print(iris.describe())

# Check for missing values
print(iris.isnull().sum())

# Pairplot
sns.pairplot(iris, hue='species')
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris.drop(columns=['species']))
plt.title('Boxplot of Iris Dataset')
plt.show()

# Violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.title('Violin Plot of Sepal Length by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.title('Violin Plot of Sepal Width by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='petal_length', data=iris)
plt.title('Violin Plot of Petal Length by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='petal_width', data=iris)
plt.title('Violin Plot of Petal Width by Species')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(8, 6))
numeric_columns = iris.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(iris[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()

