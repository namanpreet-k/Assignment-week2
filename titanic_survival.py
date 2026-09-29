import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Titanic dataset
data = pd.read_csv("train.csv")

print(data.head())

# Select the columns we need
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = data["Survived"]

# Convert male and female into numbers
X = X.copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Fill missing Age and Fare values
X["Age"] = X["Age"].fillna(X["Age"].mean())
X["Fare"] = X["Fare"].fillna(X["Fare"].mean())

# Divide the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict survival
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))