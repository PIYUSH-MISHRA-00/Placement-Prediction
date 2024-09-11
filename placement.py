import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the data
data = pd.read_csv('placement.csv')

# Data Analysis - Visualization
sns.countplot(x='placement', data=data)
plt.title('Distribution of Placement')
plt.show()

sns.pairplot(data, hue='placement')
plt.show()

# Feature Engineering: Adding interaction features
data['cgpa_iq'] = data['cgpa'] * data['iq']

# Prepare the data with the new feature
X = data[['cgpa', 'iq', 'cgpa_iq']]
y = data['placement']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Improved Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{cm}")

# Save the improved model as a pickle file
with open('placement_model_improved.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Improved model saved as 'placement_model_improved.pkl'")
