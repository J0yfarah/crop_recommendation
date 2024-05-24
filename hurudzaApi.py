# Step 1: Import necessary libraries
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Step 2: Load data
data = pd.read_csv('Crop_recommendation.csv')

# Step 3: Data preprocessing
# Forward fill missing values
data.ffill(inplace=True)

# If necessary, you can also normalize numerical features


# Encode categorical label
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the data into features (X) and target variable (y)
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# Step 7: Model deployment with Flask
app = Flask('recomend')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    # Preprocess data if neededprin
    # Assuming the data received is in the same format as the training data
    # You might need to convert it into the appropriate format for prediction
    features = [data['N'], data['P'], data['K'], data['temperature'],
                data['humidity'], data['ph'], data['rainfall']]
    # Making prediction for a single sample
    prediction = model.predict([features])[0]

    # Decode the encoded label using LabelEncoder
    predicted_crop = label_encoder.inverse_transform([prediction])[0]

    return jsonify({'predicted_crop': predicted_crop})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
