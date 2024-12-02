import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = [
    "Logistic Regression", "Decision Tree", "Random Forest",
    "K-Nearest Neighbors", "Support Vector Machine",
    "Gaussian Naive Bayes", "Multi-Layer Perceptron", "XGBoost"
]

# Accuracy scores for each model
accuracy = [87.92, 87.50, 90.83, 85.63, 88.43, 83.96, 86.25, 88.12]

# Precision, recall, and F1-score for class 0 and class 1
precision_class_0 = [0.90, 0.94, 0.93, 0.90, 0.91, 0.91, 0.89, 0.92]
recall_class_0 = [0.97, 0.92, 0.97, 0.94, 0.97, 0.91, 0.96, 0.94]
f1_class_0 = [0.93, 0.93, 0.95, 0.92, 0.94, 0.91, 0.92, 0.93]
precision_class_1 = [0.59, 0.52, 0.71, 0.42, 0.59, 0.39, 0.45, 0.56]
recall_class_1 = [0.27, 0.59, 0.51, 0.27, 0.32, 0.40, 0.22, 0.46]
f1_class_1 = [0.37, 0.55, 0.59, 0.33, 0.41, 0.39, 0.30, 0.50]

# Generate x positions for the bars
x = np.arange(len(models))

# Set the width of each bar
width = 0.2

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Plotting metrics for Class 0 (Precision, Recall, F1-Score)
ax.bar(x - width, precision_class_0, width, label="Precision (Class 0)", color='lightgreen')
ax.bar(x, recall_class_0, width, label="Recall (Class 0)", color='orange')
ax.bar(x + width, f1_class_0, width, label="F1-Score (Class 0)", color='gold')

# Plotting metrics for Class 1 (Precision, Recall, F1-Score)
ax.plot(x, precision_class_1, marker='o', label="Precision (Class 1)", color='purple', linewidth=2, linestyle='dashed')
ax.plot(x, recall_class_1, marker='s', label="Recall (Class 1)", color='red', linewidth=2, linestyle='dashed')
ax.plot(x, f1_class_1, marker='D', label="F1-Score (Class 1)", color='brown', linewidth=2, linestyle='dashed')

# Formatting the plot
ax.set_xticks(x)  # Set the x-ticks to the model names
ax.set_xticklabels(models, rotation=45, ha="right")  # Rotate the model names for readability
ax.set_title("Comparison of Metrics Across Models", fontsize=16)  # Set the title of the plot
ax.set_ylabel("Metric Value (%)", fontsize=12)  # Set the y-axis label
ax.legend(loc="best", fontsize=10)  # Display the legend in the best position
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add a dashed grid for the y-axis

# Plotting the accuracy separately
plt.figure(figsize=(12, 6))
plt.plot(models, accuracy, marker='o', linestyle='-', color='blue', label="Accuracy")

# Formatting the accuracy plot
plt.title("Accuracy of Models", fontsize=16)  # Set the title
plt.xlabel("Models", fontsize=12)  # Set the x-axis label
plt.ylabel("Accuracy (%)", fontsize=12)  # Set the y-axis label
plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-ticks for readability
plt.yticks(fontsize=10)  # Set font size for y-ticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a dashed grid for the y-axis
plt.legend(fontsize=10)  # Display the legend

# Apply tight layout to avoid overlapping elements
plt.tight_layout()

# Show the plot
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()  # Display the plots
