import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_excel('e-commerce-dataset.xlsx', sheet_name='E_Comm')
df.to_csv('e-commerce_churn.csv', index=False)
df = pd.read_csv('e-commerce_churn.csv')

##################################################################
######################## Data preparation ########################
##################################################################

print("\n---------------------------------------\n\
---------- Data Preparation -----------\n\
----------------------------------------\n")

print(df.info())
df = df.dropna()

df['Churn'] = df['Churn'].astype('category')

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

##################################################################
########################       a.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part a.) -----------------\n\
----------------------------------------\n")

# Train a random forest model
X = df.drop(columns=['Churn'])
y = df['Churn']

train_model = RandomForestClassifier(n_estimators=500, max_features=3, random_state=0)
train_model.fit(X, y)

pred = train_model.predict(X)
error_rate = np.mean(y != pred)
print(f"Error rate: {error_rate}")

# Calculate accuracy
print(f"Accuracy: {accuracy_score(y, pred)}")

##################################################################
########################       c.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part c.) -----------------\n\
----------------------------------------\n")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.20, stratify=df['Churn'], random_state=2023+2024)

# View proportions of Churn
print(f"Proportions Train:\n {train_df['Churn'].value_counts(normalize=True)}")
print(f"Proportions Test:\n {test_df['Churn'].value_counts(normalize=True)}")

# Define Model
train_model = RandomForestClassifier(n_estimators=1000, max_features=3, random_state=0)

# Cross-validation
cv_fits_accuracy = cross_val_score(train_model, X, y, cv=4, scoring='accuracy')
cv_fits_precision = cross_val_score(train_model, X, y, cv=4, scoring='precision')
cv_fits_recall = cross_val_score(train_model, X, y, cv=4, scoring='recall')

print("\nCV-Accuracy:", np.mean(cv_fits_accuracy))
print("CV-Precision:", np.mean(cv_fits_precision))
print("CV-Recall:", np.mean(cv_fits_recall))

# Train the final model
train_model.fit(train_df.drop(columns=['Churn']), train_df['Churn'])

# Variable Importance Plot
importance_values = train_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance_values})
imp_plot = importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
imp_plot.plot()
plt.show()

# Apply on test set
test_predictions = train_model.predict(test_df.drop(columns=['Churn']))
test_probabilities = train_model.predict_proba(test_df.drop(columns=['Churn']))

test_predictions_df = pd.DataFrame({'Churn': test_df['Churn'], 
                                     'Predicted_Churn': test_predictions,
                                     'Probability_Churn=0': test_probabilities[:, 0],
                                     'Probability_Churn=1': test_probabilities[:, 1]})

# Confusion Matrix
conf_matrix = confusion_matrix(test_df['Churn'], test_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

# Precision, accuracy, recall
print("\nTest-Precision:", precision_score(test_df['Churn'], test_predictions))
print("Test-Accuracy:", accuracy_score(test_df['Churn'], test_predictions))
print("Test-Recall:", recall_score(test_df['Churn'], test_predictions))