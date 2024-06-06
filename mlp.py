from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from dataprep import load_and_preprocess_data

# Loading and preprocessing the data
X_train, y_train, X_safe, y_safe = load_and_preprocess_data('./exploratory_worms_complete.csv')

# Split the remaining training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_safe = scaler.transform(X_safe)

# Multi Layer Perceptron with best found parameters using grid search
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=300, alpha=0.01,
                    solver='sgd', random_state=42, verbose=True)
mlp.fit(X_train, y_train) # training the model


# Test data prediction and evaluation
y_pred = mlp.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Safe data prediction and evaluation
safe_predictions = mlp.predict(X_safe)
print("Safe Data Accuracy:", accuracy_score(y_safe, safe_predictions))
print(classification_report(y_safe, safe_predictions))

# Plot training loss curve
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
