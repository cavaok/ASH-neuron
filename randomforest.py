from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from dataprep import load_and_preprocess_data
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting


# Loading and preprocessing the data
X, y, X_safe, y_safe = load_and_preprocess_data('./exploratory_worms_complete.csv')

# Split the remaining training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hard-coded parameters found from doing a grid search
best_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'bootstrap': True
}

# Random Forest time...
rf_classifier = RandomForestClassifier(**best_params, random_state=42) # initializes the classifier
rf_classifier.fit(X_train, y_train) # trains the model

# Test data prediction and evaluation
y_pred = rf_classifier.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Safe data prediction and evaluation
safe_predictions = rf_classifier.predict(X_safe)
print("Safe Data Accuracy:", accuracy_score(y_safe, safe_predictions))
print(classification_report(y_safe, safe_predictions))

# Doing feature importance
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 8))
feature_importances[:20].plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # most important at top of the plot
plt.show()