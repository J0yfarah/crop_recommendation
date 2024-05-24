# Step 1: Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and preprocess the dataset
# Replace 'dataset.csv' with the path to your dataset file
data = pd.read_csv('TARP.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Assuming 'status' column contains the target variable
X = data.drop('Status', axis=1)
y = data['Status']

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X_imputed, y)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model and print accuracy
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 5: Generate confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()

# Step 6: Create a Flask API for making predictions
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    moisture = data['moisture']
    temperature = data['temperature']
    humidity = data['humidity']
    # Combine features into a list
    features = [moisture, temperature, humidity]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
