# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%

import pyreadstat
import lightgbm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#%%% Importing Data

file_path = 'C:/Users/mashhadcom.com/Desktop/these/ESS-Data-Wizard-subset-2023-09-02.sav'
data, meta = pyreadstat.read_sav(file_path)


#%%
data.shape


#%%
# Define columns to keep
columns_to_keep = [
    'cntry',
    'stflife', 'stfeco', 'stfgov', 'stfdem', 'stfedu', 'stfhlth',
    'happy', 'health',
    'rlgdgr', 'ctzcntr', 'brncntr',
    'implvdm',
    'gndr', 'agea', 'eisced', 'hincfel',
    'atcherp', 'ppltrst', 'trstep',  # targets
    'trstlgl', 'trstplc', 'trstplt', 'trstprt', 'trstun', 'trstprl' 
]

data01 = data[columns_to_keep]

data01.head()


#%% check the missing values 
missing_percentages01 = (data01.isnull().sum() / len(data01)) * 100
print("Percentage of missing values in each column:")
print(missing_percentages01)



#%% mapping the missing values
data01['agea'] = data01['agea'].replace(999, np.nan)



# Define the label mapping dictionary
label_mapping = {
'stflife': {77: np.nan, 88: np.nan, 99: np.nan},
'stfeco': {77: np.nan, 88: np.nan, 99: np.nan},
'stfgov': {77: np.nan, 88: np.nan, 99: np.nan},
'stfdem': {77: np.nan, 88: np.nan, 99: np.nan},
'stfedu': {77: np.nan, 88: np.nan, 99: np.nan},
'stfhlth': {77: np.nan, 88: np.nan, 99: np.nan},
'happy': {77: np.nan, 88: np.nan, 99: np.nan},
'health': {7: np.nan, 8: np.nan, 9: np.nan},
'rlgdgr': {77: np.nan, 88: np.nan, 99: np.nan},
'ctzcntr': {7: np.nan, 8: np.nan, 9: np.nan},
'brncntr': {7: np.nan, 8: np.nan, 9: np.nan},
'implvdm': {77: np.nan, 88: np.nan, 99: np.nan},
'gndr': {9: np.nan},
'eisced': {55: np.nan,77: np.nan, 88: np.nan, 99: np.nan},
'hincfel': {7: np.nan, 8: np.nan, 9: np.nan},
'trstlgl': {77: np.nan, 88: np.nan, 99: np.nan},
'trstplc': {77: np.nan, 88: np.nan, 99: np.nan},
'trstplt': {77: np.nan, 88: np.nan, 99: np.nan},
'trstprt': {77: np.nan, 88: np.nan, 99: np.nan},
'trstun': {77: np.nan, 88: np.nan, 99: np.nan},
'atcherp': {77: np.nan, 88: np.nan, 99: np.nan},
'ppltrst': {77: np.nan, 88: np.nan, 99: np.nan},
'trstep': {77: np.nan, 88: np.nan, 99: np.nan},
'trstprl': {77: np.nan, 88: np.nan, 99: np.nan},
}

# Apply label mapping to specified columns
for feature, mapping in label_mapping.items():
    data01[feature] = data01[feature].replace(mapping)

  

#%% checking Missing values
missing_percentages01 = (data01.isnull().sum() / len(data01)) * 100
print("Percentage of missing values in each column:")
print(missing_percentages01)



#%% imputing missing values 
from sklearn.impute import SimpleImputer

features = [
  'agea', 
  'stflife',
  'stfeco',
  'stfgov',
  'stfdem',
  'stfedu',
  'stfhlth',
  'happy',
  'health',
  'rlgdgr',
  'ctzcntr',
  'brncntr',
  'implvdm',
  'gndr',
  'eisced',
  'hincfel',
  'trstlgl',
  'trstplc',
  'trstplt',
  'trstprt',
  'trstun',
  'atcherp',
  'ppltrst',
  'trstep',
  'trstprl'
]

imputer = SimpleImputer(strategy='median')
data01[features] = imputer.fit_transform(data01[features])




#%% check missing values
missing_percentagesData01 = (data01.isnull().sum() / len(data01)) * 100
print("Percentage of missing values in each column:")
print(missing_percentagesData01)

#%% encoding 
data02 = pd.get_dummies(data01, columns=['cntry'], drop_first=True)

categorical_features = [
    'stflife',
    'stfeco',
    'stfgov',
    'stfdem',
    'stfedu',
    'stfhlth',
    'happy',
    'health',
    'rlgdgr',
    'ctzcntr',
    'brncntr',
    'implvdm',
    'gndr',
    'eisced',
    'hincfel',
    'trstlgl',
    'trstplc',
    'trstplt',
    'trstprt',
    'trstun',
    'atcherp',
    'ppltrst',
    'trstep',
]

encoder = LabelEncoder()
for feature in categorical_features:
    original_values = data02[feature].unique()
    data[feature] = encoder.fit_transform(data02[feature])
    encoded_values = data02[feature].unique()
    print(f"Feature: {feature}")
    print("Original Values:", original_values)
    print("Encoded Values:", encoded_values)
    print("-------------------------------------")


#%% constructing target factor

# Calculate resiliance
data02['resilience_bin'] = (data02[['trstep', 'atcherp', 'ppltrst']].mean(axis=1) > 5).astype(int)
data021 = data02.drop(['trstep', 'atcherp', 'ppltrst'], axis=1)
# Display df3
print(data021)


# Save the data021 DataFrame to a CSV file
pyreadstat.write_sav(data021, 'data021.sav')

#%% plots
import matplotlib.pyplot as plt
import seaborn as sns

# Select the variables 
selected_variables = [
    'agea',
    'stflife',
    'stfeco',
    'stfgov',
    'stfdem',
    'stfedu',
    'stfhlth',
    'happy',
    'health',
    'rlgdgr',
    'implvdm',
    'gndr',
    'eisced',
    'hincfel',
    'trstlgl',
    'trstplc',
    'trstplt',
    'trstprt',
    'trstun',
    'resilience_bin'
]

# Set the number of rows and columns for subplots
num_rows = 4  
num_cols = 5  

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
fig.subplots_adjust(hspace=0.5)  


axes = axes.ravel()

# Create a dictionary to map variable names to labels
variable_labels = {
    'stflife': 'Life Satisfaction',
    'stfeco': 'Economic Situation',
    'stfgov': 'Government Trust',
    'stfdem': 'Democracy Trust',
    'stfedu': 'Education Trust',
    'stfhlth': 'Health Trust',
    'happy': 'Happiness',
    'health': 'Self-Reported Health',
    'rlgdgr': 'Religious Degree',
    'implvdm': 'Implementation of Values',
    'gndr': 'Gender',
    'eisced': 'Education Level',
    'hincfel': 'Household Income',
    'trstlgl': 'Trust in Legal System',
    'trstplc': 'Trust in Police',
    'trstplt': 'Trust in Political Parties',
    'trstprt': 'Trust in Parliament',
    'trstun': 'Trust in United Nations',
    'agea': 'Age',
    'resilience_bin': 'Resilience'
}

# Loop through the selected variables and create histograms without density curves
for i, variable in enumerate(selected_variables):
    ax = axes[i]
    if variable in categorical_features:
        # For categorical variables, use a countplot
        sns.countplot(data=data02, x=variable, ax=ax)
        ax.set_title(f'Countplot of {variable_labels[variable]}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  
    else:
        # For numerical variables, use a histogram without density curve
        sns.histplot(data=data02, x=variable, ax=ax, kde=False, bins=20)  
        ax.set_title(f'Histogram of {variable_labels[variable]}')
    
plt.tight_layout()
plt.show()

#%% summary statistics

# Variable labels
variable_labels = {
    'stflife': 'Life Satisfaction',
    'stfeco': 'Economic Situation',
    'stfgov': 'Government Trust',
    'stfdem': 'Democracy Trust',
    'stfedu': 'Education Trust',
    'stfhlth': 'Health Trust',
    'happy': 'Happiness',
    'health': 'Self-Reported Health',
    'rlgdgr': 'Religious Degree',
    'implvdm': 'Implementation of Values',
    'gndr': 'Gender',
    'eisced': 'Education Level',
    'hincfel': 'Household Income',
    'trstlgl': 'Trust in Legal System',
    'trstplc': 'Trust in Police',
    'trstplt': 'Trust in Political Parties',
    'trstprt': 'Trust in Parliament',
    'trstun': 'Trust in United Nations',
    'agea': 'Age',
    'resilience_bin': 'Resilience'
}

# Compute summary statistics
summary_stats = data02.describe()
#%%  correlations target_correlations barplot  selected_variables

# Compute correlations with the target variable ('resilience_bin')
target_correlations = data02[selected_variables].corr()['resilience_bin'].drop('resilience_bin')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlations.values, y=target_correlations.index, palette='viridis')
plt.title("Correlation with Resilience (Target)")
plt.xlabel("Correlation Coefficient")
plt.show()

#%% heatmap

import seaborn as sns
import matplotlib.pyplot as plt



# Compute correlations between selected variables
correlation_matrix = data02[selected_variables].corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()




#%% correlations barplot  total_variables

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlations with the target variable ('resilience_bin')
target_correlations = data02.corr()['resilience_bin'].drop('resilience_bin')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlations.values, y=target_correlations.index, palette='viridis')
plt.title("Correlation with Resilience (Target)")
plt.xlabel("Correlation Coefficient")
plt.show()



#%% Split the data 
import random
from sklearn.model_selection import train_test_split

# Make a copy of DataFrame
data03 = data021.copy() # for models
data04_3d = data021.copy() # for 3d loss surface

# Set the random seed
random.seed(42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data03.drop(['resilience_bin'], axis=1), data03['resilience_bin'], test_size=0.2, random_state=42)

#%% DummyClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score



# Define the dummy classifier 
dummy_model = DummyClassifier(strategy='most_frequent')

# Train the dummy classifier (no actual training, as it uses simple rules)
dummy_model.fit(X_train, y_train)

# Make predictions on the test set
dummy_predictions = dummy_model.predict(X_test)

# Calculate performance metrics for the dummy classifier
dummy_confusion_matrix = confusion_matrix(y_test, dummy_predictions)
dummy_accuracy = accuracy_score(y_test, dummy_predictions)
dummy_recall = recall_score(y_test, dummy_predictions)
dummy_f1_score = f1_score(y_test, dummy_predictions, average='weighted')

# Print the report for the dummy classifier (baseline)
print("Dummy Classifier (Baseline) Report:")
print(f'Accuracy: {dummy_accuracy:.3f}')
print(f'Recall: {dummy_recall:.3f}')
print(f'F1-score: {dummy_f1_score:.3f}')
print("Confusion Matrix:")
print(dummy_confusion_matrix)

#%% DummyClassifier ConfusionMatrixDisplay

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=dummy_confusion_matrix, display_labels=dummy_model.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Dummy Model Confusion Matrix")
plt.show()
#%% DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Define the base Decision Tree model
base_model = DecisionTreeClassifier() 

# Train the base Decision Tree model
base_model.fit(X_train, y_train)

# Make predictions on the test set
base_predictions = base_model.predict(X_test)

# Calculate performance metrics for the base model
base_confusion_matrix = confusion_matrix(y_test, base_predictions)
base_accuracy = accuracy_score(y_test, base_predictions)
base_recall = recall_score(y_test, base_predictions)
base_f1_score = f1_score(y_test, base_predictions, average='weighted')

# Print the report for the base model
print("Base Decision Tree Model Report:")
print(f'Accuracy: {base_accuracy:.3f}')
print(f'Recall: {base_recall:.3f}')
print(f'F1-score: {base_f1_score:.3f}')
print("Confusion Matrix:")
print(base_confusion_matrix)

#%% DecisionTreeClassifier ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=base_confusion_matrix, display_labels=base_model.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Base Model Confusion Matrix")
plt.show()

#%% LogisticRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score



# Define the Logistic Regression model
logistic_model = LogisticRegression()

# Define parameter grid for grid search

logistic_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],  # Type of regularization penalty (L1 or L2)
    'max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
}

# Create a GridSearchCV object for Logistic Regression with cross-validation
logistic_grid_search = GridSearchCV(logistic_model, param_grid=logistic_params, cv=5, scoring='f1_weighted')

# Train the Logistic Regression model
logistic_grid_search.fit(X_train, y_train)

# Make predictions on the test set
logistic_predictions = logistic_grid_search.predict(X_test)

# Calculate performance metrics
logistic_confusion_matrix = confusion_matrix(y_test, logistic_predictions)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_recall = recall_score(y_test, logistic_predictions)
logistic_f1_score = f1_score(y_test, logistic_predictions, average='weighted')

# Print the report for Logistic Regression
print("Logistic Regression Report:")
print(f'Accuracy: {logistic_accuracy:.3f}')
print(f'Recall: {logistic_recall:.3f}')
print(f'F1-score: {logistic_f1_score:.3f}')
print("Confusion Matrix:")
print(logistic_confusion_matrix)
print("Best model parameters:", logistic_grid_search.best_params_)


#%% LogisticRegression ConfusionMatrixDisplay
# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=logistic_confusion_matrix, display_labels=logistic_grid_search.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

#%% RandomForestClassifier

# Create a copy of data021 for subsequent models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# Define the Random Forest model
rf_model = RandomForestClassifier()

# Define parameter grid for grid search


rf_params = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],  # Maximum tree depth (None means no limit)
}


# Create a GridSearchCV object for Random Forest with cross-validation
rf_grid_search = GridSearchCV(rf_model, param_grid=rf_params, cv=5, scoring='f1_weighted')

# Train the Random Forest model on the training data
rf_grid_search.fit(X_train, y_train)  

# Make predictions on the test set
rf_predictions = rf_grid_search.predict(X_test)  

# Calculate performance metrics
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)  
rf_accuracy = accuracy_score(y_test, rf_predictions)  
rf_recall = recall_score(y_test, rf_predictions) 
rf_f1_score = f1_score(y_test, rf_predictions, average='weighted')  

# Print the report for Random Forest
print("Random Forest Report:")
print(f'Accuracy: {rf_accuracy:.3f}')
print(f'Recall: {rf_recall:.3f}')
print(f'F1-score: {rf_f1_score:.3f}')
print("Confusion Matrix:")
print(rf_confusion_matrix)
#%% RandomForestClassifier  Feature Importance
# Feature Importance Report
rf_feature_importance = rf_grid_search.best_estimator_.feature_importances_
sorted_idx = rf_feature_importance.argsort()[::-1]

# Print Feature Importance
print("Feature Importance:")
for i, idx in enumerate(sorted_idx):
    feature = data021.drop(['resilience_bin'], axis=1).columns[idx] 
    importance = rf_feature_importance[idx]
    print(f"{i + 1}. {feature}: {importance:.3f}")

#%% RandomForestClassifier ConfusionMatrixDisplay

# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=rf_confusion_matrix, display_labels=rf_grid_search.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Random Forest")
plt.show()


#%% main model > LGBMClassifier
# Define the LightGBM model
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier()

# Define parameter grid for grid search

lgbm_params = {
    'num_leaves': [31, 63, 127],
    'max_depth': [5, 10, 15],  # Maximum tree depth
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'n_estimators': [50, 300, 500],  # Number of boosting rounds
}


# Create a GridSearchCV object for LightGBM with cross-validation
lgbm_grid_search = GridSearchCV(lgbm_model, param_grid=lgbm_params, cv=5, scoring='f1_weighted')

# Train the LightGBM model on the training data
lgbm_grid_search.fit(X_train, y_train)  

# Make predictions on the test set
lgbm_predictions = lgbm_grid_search.predict(X_test)  

# Calculate performance metrics
lgbm_confusion_matrix = confusion_matrix(y_test, lgbm_predictions)  
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)  
lgbm_recall = recall_score(y_test, lgbm_predictions) 
lgbm_f1_score = f1_score(y_test, lgbm_predictions, average='weighted')  

# Print the report for LightGBM
print("LightGBM Report:")
print(f'Accuracy: {lgbm_accuracy:.3f}')
print(f'Recall: {lgbm_recall:.3f}')
print(f'F1-score: {lgbm_f1_score:.3f}')
print("Confusion Matrix:")
print(lgbm_confusion_matrix)
print("Best model parameters:", lgbm_grid_search.best_params_)

#%% second model with 3d log loss  %matplotlib qt
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets for 3d loss surface
X_train_3d, X_valid_3d, y_train_3d, y_valid_3d = train_test_split(data04_3d.drop(['resilience_bin'], axis=1), data04_3d['resilience_bin'], test_size=0.2, random_state=42)

# Define a function to calculate loss for given hyperparameters
def calculate_loss_3d(learning_rate, n_estimators):
    # Create and train a LightGBM classifier
    lgbm_model_3d = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=10, num_leaves=31)
    lgbm_model_3d.fit(X_train_3d, y_train_3d)

    # Make predictions on the validation set
    y_pred_3d = lgbm_model_3d.predict(X_valid_3d)

    # Calculate log loss
    loss_3d = log_loss(y_valid_3d, y_pred_3d)

    return loss_3d

# Define a range of hyperparameters to explore base on the main model parameters
learning_rates_3d = np.linspace(0.01, 0.11, 10)
n_estimators_values_3d = np.linspace(50, 500, 20, dtype=int)

# Create a grid of hyperparameters
learning_rate_grid_3d, n_estimators_grid_3d = np.meshgrid(learning_rates_3d, n_estimators_values_3d)

# Calculate loss for each combination of hyperparameters
loss_grid_3d = np.zeros_like(learning_rate_grid_3d, dtype=float)
for i in range(learning_rate_grid_3d.shape[0]):
    for j in range(learning_rate_grid_3d.shape[1]):
        loss_grid_3d[i, j] = calculate_loss_3d(learning_rate_grid_3d[i, j], n_estimators_grid_3d[i, j])

# Enable interactive mode for Matplotlib
plt.ion()

# Create the 3D plot
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.plot_surface(learning_rate_grid_3d, n_estimators_grid_3d, loss_grid_3d, cmap='viridis')
ax_3d.set_xlabel('Learning Rate')
ax_3d.set_ylabel('Number of Estimators')
ax_3d.set_zlabel('Log Loss')
ax_3d.set_title('3D Loss Surface')

plt.show()

# Find indices of minimum loss
min_loss_index = np.unravel_index(np.argmin(loss_grid_3d), loss_grid_3d.shape)

# Get the corresponding learning rate and number of estimators for the minimum loss
best_learning_rate = learning_rate_grid_3d[min_loss_index]
best_n_estimators = n_estimators_grid_3d[min_loss_index]

# Minimum log loss
min_log_loss = np.min(loss_grid_3d)

print(f"Minimum Log Loss: {min_log_loss}")
print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best Number of Estimators: {best_n_estimators}")


#%% best_lgbm_model & Plot the best tree & trees_to_dataframe
from lightgbm import LGBMClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# Get the best estimator from the grid search
best_lgbm_model = lgbm_grid_search.best_estimator_

# Make predictions on the test set using the best model
lgbm_predictions = best_lgbm_model.predict(X_test)

# Calculate performance metrics
lgbm_confusion_matrix = confusion_matrix(y_test, lgbm_predictions)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
lgbm_recall = recall_score(y_test, lgbm_predictions)
lgbm_f1_score = f1_score(y_test, lgbm_predictions, average='weighted')

# Print the report
print("Performance Metrics:")
print(f'Accuracy: {lgbm_accuracy:.3f}')
print(f'Recall: {lgbm_recall:.3f}')
print(f'F1-score: {lgbm_f1_score:.3f}')
print("Confusion Matrix:")
print(lgbm_confusion_matrix)
print("Best model parameters:", lgbm_grid_search.best_params_)


# Get the best iteration (tree) index based on validation data
best_iteration = best_lgbm_model.best_iteration_

# Plot the best tree
plot_tree(best_lgbm_model, tree_index=best_iteration)
plt.show()


#second tree
graph = lightgbm.create_tree_digraph(best_lgbm_model, tree_index=best_iteration, name=f'Tree{best_iteration}')
graph.graph_attr.update(size="110,110")
graph


#trees data fram
# Assuming best_lgbm_model is the trained LGBMClassifier model
booster = best_lgbm_model.booster_

#the trees_to_dataframe method
tree_df = booster.trees_to_dataframe()


sampled_tree_df = tree_df.sample(n=15, random_state=42)
sampled_tree_df.to_excel('sampled_tree_df .xlsx', index=False)

#%% MLPClassifier()
from sklearn.neural_network import MLPClassifier

# Define the Neural Network (NN) model
nn_model = MLPClassifier()

# Define parameter grid for grid search
nn_params = {
    'hidden_layer_sizes': [(20, 10, 4), (50, 25, 4)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
}

# Create a GridSearchCV object for the NN with cross-validation
nn_grid_search = GridSearchCV(nn_model, param_grid=nn_params, cv=5, scoring='f1_weighted')

# Train the NN model on the training data 
nn_grid_search.fit(X_train, y_train)  

# Make predictions on the test set
nn_predictions = nn_grid_search.predict(X_test)  

# Calculate performance metrics 
nn_confusion_matrix = confusion_matrix(y_test, nn_predictions) 
nn_accuracy = accuracy_score(y_test, nn_predictions)  
nn_recall = recall_score(y_test, nn_predictions)  
nn_f1_score = f1_score(y_test, nn_predictions, average='weighted') 

# Print the report for the NN
print("Neural Network (NN) Report:")
print(f'Accuracy: {nn_accuracy:.3f}')
print(f'Recall: {nn_recall:.3f}')
print(f'F1-score: {nn_f1_score:.3f}')
print("Confusion Matrix:")
print(nn_confusion_matrix)
print("Best model parameters:", nn_grid_search.best_params_)

#%% MLPClassifier ConfusionMatrixDisplay
# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=nn_confusion_matrix, display_labels=nn_grid_search.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Neural Network (NN)")
plt.show()

#%% best_lgbm_model - feature importances - Precision-Recall curve -Confusion Matrix Display

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split


# Extract feature importances from the trained LightGBM model
importances = best_lgbm_model.feature_importances_
feature_names = X_train.columns  

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (LightGBM)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# Generate predictions from the LightGBM model
y_pred_proba = best_lgbm_model.predict_proba(X_test)[:, 1]

# Calculate precision and recall values using precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)





# Create and plot the Confusion Matrix Display
cm_display = ConfusionMatrixDisplay(confusion_matrix=lgbm_confusion_matrix, display_labels=best_lgbm_model.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for LightGBM")
plt.show()





#%% shap
import shap


X1 = X_train  # Drop the target column from X
x_interest1 = X1.iloc[0].values.reshape(1, -1)  # Reshape to make it 2-dimensional

# Wrap the LightGBM model using SHAP's TreeExplainer
explainer = shap.TreeExplainer(best_lgbm_model)
shap_values = explainer.shap_values(x_interest1)
shap.summary_plot(shap_values, X1)

shap.plots.bar(shap_values)

#%% LaTeX table

# Create a DataFrame with model names, F1 scores, and accuracy scores
model_names = ['Dummy Model', 'Base Model', 'Logistic Regression', 'Random Forest', 'LightGBM', 'Neural Network']
f1_scores = [dummy_f1_score, base_f1_score, logistic_f1_score, rf_f1_score, lgbm_f1_score, nn_f1_score]
accuracy_scores = [dummy_accuracy, base_accuracy, logistic_accuracy, rf_accuracy, lgbm_accuracy, nn_accuracy]

# Create a new DataFrame
result_df = pd.DataFrame({'Model': model_names, 'F1 Score': f1_scores, 'Accuracy': accuracy_scores})

# Sort the DataFrame by F1 Score and Accuracy in descending order
result_df = result_df.sort_values(by=['F1 Score', 'Accuracy'], ascending=False)

# Convert the DataFrame to LaTeX format
latex_table_latex = result_df.to_latex(index=False, float_format="%.3f", escape=False, column_format='lll')

# Print the LaTeX table
print(latex_table_latex)

#%%
print("LightGBM Best model parameters:", lgbm_grid_search.best_params_)
print("Neural Network Best model parameters:", nn_grid_search.best_params_)
print("Random Forest Best model parameters:", rf_grid_search.best_params_)
print("Logistic Regression Best model parameters:", logistic_grid_search.best_params_)

