# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from a CSV file
data = pd.read_csv('winequality-red.csv')

# Display the first 5 rows of the dataset to get a glimpse of the data
data.head()

# Print the column names in the dataset
print("Column headers/names: {}".format(list(data.columns)))

# Print the shape (number of rows and columns) of the dataset
print("Shape of Red Wine dataset: {}".format(data.shape))

# Print the statistical summary of the dataset (mean, std, min, max, etc.)
data.describe()

# Display concise summary of the dataset, including data types and non-null values
data.info()

# Check for missing values in the dataset (returns a count of null values for each column)
data.isnull().sum()

# Display the frequency of each wine quality value (number of wines of each quality rating)
data['quality'].value_counts()

# Create a countplot to visualize the distribution of wine quality ratings
sns.countplot(data=data, x='quality')

# Add title and labels to the plot for better understanding
plt.title('Count of Wines by Quality')
plt.xlabel('Quality')
plt.ylabel('Count')

# Show the countplot
plt.show()

# Create a histogram with a Kernel Density Estimate (KDE) for the 'alcohol' feature
plt.figure(figsize=(10, 6))
sns.histplot(data['alcohol'], kde=True)  # kde=True adds a kernel density estimate to the histogram

# Add title and labels to the plot
plt.title('Distribution of Alcohol Percentage')
plt.xlabel('Alcohol Percentage')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Create box plots for each feature in the dataset to visualize the distribution and detect outliers
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, figsize=(12, 10))

# Set a title for the box plot figure
plt.suptitle('Box Plots for Each Feature in the Dataset', fontsize=16)

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

# Show the box plot
plt.show()

# Create density plots for each feature in the dataset to visualize the distribution
data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, figsize=(12, 10))

# Set a title for the density plot figure
plt.suptitle('Density Plots for Each Feature in the Wine Dataset', fontsize=16)

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

# Show the density plot
plt.show()

# Create histograms for each feature in the dataset to visualize their distributions
data.hist(figsize=(10, 10), bins=50)

# Set a title for the histogram figure
plt.suptitle('Histograms for Each Feature in the Dataset', fontsize=16)

# Show the histograms
plt.show()

# Calculate and display the mean of all features grouped by the 'quality' of the wine
data.groupby('quality').mean()
