import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Fertilizer Prediction.csv')

# Drop Soil Type and Crop Type columns
data = data.drop(columns=['Soil Type', 'Crop Type'])

# Preprocess the data
data = data.dropna()  # Drop rows with missing values if any

# Encode categorical columns
label_encoder = LabelEncoder()
data['Fertilizer Name'] = label_encoder.fit_transform(data['Fertilizer Name'])

# Define features and target
X = data.drop(columns='Fertilizer Name')
y = data['Fertilizer Name']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train different models
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

# Train and evaluate the models
best_model = None
best_accuracy = 0
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Load the trained model and label encoder
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features from the request
    features = np.array([data['Temperature'], data['Humidity'], data['Moisture'],
                         data['Nitrogen'], data['Potassium'], data['Phosphorous']]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    fertilizer_name = label_encoder.inverse_transform(prediction)[0]

    # Return the prediction as a JSON response
    # Changed key to 'prediction'
    return jsonify({'prediction': fertilizer_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
