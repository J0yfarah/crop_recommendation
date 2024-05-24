import requests

# Define the URL of your API endpoint
# Assuming your Flask app is running locally
url = 'http://127.0.0.1:5000/predict'

# Sample data for prediction
data = {
    'N': 17,
    'P': 19,
    'K': 55,
    'temperature': 28,
    'humidity': 11,
    'ph': 6,
    'rainfall': 50
}

# Send POST request
response = requests.post(url, json=data)

# Print the prediction
print(response.json())
