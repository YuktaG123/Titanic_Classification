import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load and preprocess the data
# Replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('/content/titanic.csv')

# Drop any irrelevant columns and handle missing values
data = data.dropna(subset=['Age', 'Sex', 'Survived'])

# Convert 'Sex' categorical variable to numerical using LabelEncoder
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Step 2: Split the data into training and testing sets
X = data[['Age', 'Sex']]  # Replace with relevant features
y = data['Survived']  # Replace with the target label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
