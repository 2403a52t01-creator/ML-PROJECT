import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("Housing.csv")

# Example: predicting price (change if needed)
X = data.drop("price", axis=1)
y = data["price"]

# Convert categorical data
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save columns
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained and saved!")