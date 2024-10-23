# cmse802_project
## Unlocking the Secrets of High-Quality Wine Using Machine Learning ##

### Brief Description:
In this project, I will work with Kaggle’s Red Wine Quality dataset to create several classification models designed to predict if a particular red wine is considered “good quality.” Each wine is given a quality score between 0 and 10. For this analysis, I converted the scores into a binary classification: wines with a score of 7 or higher are categorized as “good quality,” while those with scores below 7 are classified as not good quality.

The quality of the wine is assessed based on 11 input variables:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulfates
* Alcohol

### Objectives:
The aims of this project include:

* Experimenting with various classification techniques to identify which one achieves the highest accuracy.
* Determining which features are most predictive of good quality wine.

Steps Involved in the Project:
1. Importing Libraries
2. Loading the Data
3. Understanding the Data
4. Addressing Missing Values
5. Exploring Variables (Data Analysis)
6. Selecting Features
7. Analyzing the Proportion of Good vs. Bad Wines
8. Preparing Data for Modeling
9. Applying Different Models
10. Selecting the Best Model

### Set up the Environment

Install Jupyter Notebook:
  pip install notebook

Install Required Libraries:
* pip install numpy pandas matplotlib seaborn scikit-learn
* pip install -r requirements.txt

Clone or Download the Project Repository:
  git clone https://github.com/yourusername/yourproject.git

Open Jupyter Notebook
  To launch Jupyter Notebook, follow these steps:
* cd path_to_your_project_folder
* jupyter notebook
  A web browser window will open, displaying the Jupyter interface. Navigate to your notebook file (e.g., cmse802_project.ipynb) and open it.

Run the Code:
  You can run individual cells of code by selecting the cell and pressing Shift + Enter.
  Alternatively, you can run all cells by selecting Cell > Run All from the menu bar.

Understanding the Notebook Structure:
  The notebook is divided into different sections, typically in this order:

1. Data Loading: This section loads the red wine dataset.
2. Data Exploration: Visualizations like histograms, box plots, and scatter plots are generated to analyze patterns in the data.
3. Feature Engineering: Feature extraction and transformations such as encoding of labels.
4. Model Training: Training of different machine learning models (e.g., Logistic Regression, Decision Trees).
5. Model Evaluation: Accuracy, confusion matrices, and other metrics to evaluate the models.
