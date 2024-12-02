# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

# Examining the variation of fixed acidity in different qualities of wine using a scatter plot
plt.scatter(data['quality'], data['fixed acidity'], label='Fixed Acidity')  # Scatter plot for 'quality' vs 'fixed acidity'
plt.title('Variation of Fixed Acidity in Different Qualities of Wine')  # Adding title
plt.xlabel('Quality')  # Label for the x-axis
plt.ylabel('Fixed Acidity')  # Label for the y-axis
plt.legend()  # Display legend
plt.show()  # Show the plot

# Create a violin plot to visualize the distribution of alcohol content for each wine quality
plt.figure(figsize=(10, 6))  # Set figure size
sns.violinplot(x='quality', y='alcohol', data=data)  # Violin plot of alcohol content by quality

# Set labels and title
plt.xlabel('Wine Quality')
plt.ylabel('Alcohol Content')
plt.title('Alcohol Content by Wine Quality')

# Show the plot
plt.show()

# Create a bar chart to show the relationship between alcohol content and wine quality
plt.figure(figsize=(10, 6))  # Set figure size
plt.bar(data['quality'], data['alcohol'], label='Alcohol Content')  # Bar plot of alcohol content by quality
plt.title('Alcohol Content by Wine Quality')  # Add title
plt.xlabel('Quality')  # Add x-axis label
plt.ylabel('Alcohol')  # Add y-axis label
plt.legend()  # Show legend
plt.show()  # Display the plot

# Create a bar plot for citric acid content by wine quality, with hue based on 'quality'
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='quality', y='citric acid', data=data, hue='quality', legend=False)  # Bar plot of citric acid by quality

# Add title and axis labels
plt.title('Citric Acid Levels by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Citric Acid')

# Show the plot
plt.show()

# Create a bar plot for residual sugar by wine quality
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='quality', y='residual sugar', data=data)  # Bar plot of residual sugar by quality

# Add title and axis labels
plt.title('Residual Sugar Levels by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Residual Sugar')

# Show the plot
plt.show()

# Create a bar plot for chloride content by wine quality
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='quality', y='chlorides', data=data)  # Bar plot of chlorides by quality

# Add title and labels
plt.title('Chloride Levels by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Chlorides')

# Show the plot
plt.show()

# Create a bar plot for free sulfur dioxide by wine quality
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='quality', y='free sulfur dioxide', data=data)  # Bar plot of free sulfur dioxide by quality

# Add title and labels
plt.title('Free Sulfur Dioxide Levels by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Free Sulfur Dioxide')

# Show the plot
plt.show()

# Create a bar plot for sulphates by wine quality
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='quality', y='sulphates', data=data)  # Bar plot of sulphates by quality

# Add title and labels
plt.title('Sulphate Levels by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Sulphates')

# Show the plot
plt.show()

# Create a figure and axis for the correlation heatmap
f, ax = plt.subplots(figsize=(10, 8))

# Calculate the correlation matrix of the features
corr = data.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr, 
            mask=np.zeros_like(corr, dtype=bool),  # No mask
            cmap=sns.diverging_palette(220, 10, as_cmap=True),  # Color palette
            square=True, 
            annot=True,  # Display correlation values in cells
            fmt=".2f",  # Format for the annotations
            cbar_kws={"shrink": .8},  # Shrink color bar
            ax=ax)

# Add title to the heatmap
plt.title('Correlation Heatmap of Wine Features')

# Save and show the plot
plt.savefig("Fig1.jpg")  # Save the heatmap as an image
plt.show()  # Display the heatmap

# Display the correlation matrix
data.corr()

# Display correlations with 'quality' sorted in descending order
corr['quality'].sort_values(ascending=False)

# Create a pairplot of all features in the dataset
sns.pairplot(data)

# Save the pairplot as an image
plt.savefig("Fig.jpg")

# Drop columns that are not significantly correlated with 'quality'
data = data.drop(['volatile acidity', 'total sulfur dioxide', 'density', 'chlorides'], axis=1)

# Create a new column 'rating' based on quality, dividing wines into 'Good' and 'Bad'
conditions = [
    (data['quality'] >= 7),  # Good wine if quality is 7 or higher
    (data['quality'] < 7)    # Bad wine if quality is less than 7
]
rating = ['Good', 'Bad']  # Labels for the ratings
data['rating'] = np.select(conditions, rating)  # Apply the conditions to create 'rating'

# Check the distribution of the new 'rating' column
data.rating.value_counts()

# Create a LabelEncoder instance for transforming 'rating' into numerical values
le = LabelEncoder()

# Transform the 'rating' column
data['rating'] = le.fit_transform(data['rating'])

# Display the counts of transformed 'rating' values
quality_counts = data['rating'].value_counts()
print(quality_counts)

# Group data by 'rating' and calculate the mean of each feature
data.groupby('rating').mean()

# Separate the features (X) and the target variable (Y) for modeling
X = data.drop(['rating', 'quality'], axis=1)
Y = data['rating']

# Create box plots for 'alcohol', 'sulphates', and 'fixed acidity' by wine ratings
bx = sns.boxplot(x="rating", y='alcohol', data=data)
bx.set(xlabel='Wine Ratings', ylabel='Alcohol Percentage', title='Alcohol Percentage in Different Wine Ratings')
plt.show()

bx = sns.boxplot(x="rating", y='sulphates', data=data)
bx.set(xlabel='Wine Ratings', ylabel='Sulphates', title='Sulphates in Different Types of Wine Ratings')
plt.show()

bx = sns.boxplot(x="rating", y='fixed acidity', data=data)
bx.set(xlabel='Wine Ratings', ylabel='Fixed Acidity', title='Fixed Acidity in different types of Wine ratings')
plt.show()

# Create a violin plot for citric acid by wine ratings
bx = sns.violinplot(x="rating", y='citric acid', data=data)
bx.set(xlabel='Wine Ratings', ylabel='Citric Acid', title='Citric Acid in Different Types of Wine Ratings')
plt.show()

# Create a strip plot for pH levels by wine ratings
bx = sns.stripplot(x="rating", y="pH", data=data, jitter=True, color='black')

# Optionally, overlay with a swarm plot
bx = sns.swarmplot(x="rating", y="pH", data=data, alpha=0.6)

# Set labels and title for pH plot
bx.set(xlabel='Wine Ratings', ylabel='pH', title='pH Levels in Different Types of Wine Ratings')
plt.show()

# Create and fit the Extra Trees Classifier to predict wine ratings based on features
classifiern = ExtraTreesClassifier(random_state=1)  # Set random_state for reproducibility
classifiern.fit(X, Y)  # Fit the classifier to the data

# Get and display feature importances (how much each feature contributes to the model)
score = classifiern.feature_importances_
print(score)
