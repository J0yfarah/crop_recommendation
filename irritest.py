import requests

# Define the URL of your Flask API endpoint
url = 'http://127.0.0.1:5001/suggest'

# Define the features for prediction
features = {
    # Example values for Soil Moisture, Temperature, Soil Humidity
    'features': [90, 20, 30]
}

# Send a POST request to the Flask API
response = requests.post(url, json=features)

# Check if the request was successful
if response.status_code == 200:
    # Print the predicted result
    print("irrigation Should be:", response.json()['prediction'])
else:
    print("Error:", response.text)
