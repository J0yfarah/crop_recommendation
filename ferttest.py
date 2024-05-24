import requests

# URL of the Flask API endpoint
url = "http://10.42.0.34:5002/predict"

# JSON payload with the features
payload = {
    "Temperature": 23,
    "Humidity": 80,
    "Moisture": 50,
    "Nitrogen": 10,
    "Potassium": 20,
    "Phosphorous": 15
}

# Send a POST request to the API
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print("Fertilizer Prediction:", response.json())
else:
    print("Failed to get prediction. Status code:", response.status_code)
    print("Response:", response.text)
