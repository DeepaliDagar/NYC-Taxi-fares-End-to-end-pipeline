import requests

api_url = "https://hbyybpply7.execute-api.us-east-2.amazonaws.com/default/TaxiPredictionFunction"
csv_data = "40:00.0,2010-03-14 15:40:00 UTC,-73.979872,40.749027,-73.976553,40.757498,5,15,3,6,1.0"

headers = {'Content-Type': 'text/csv'}

response = requests.post(api_url, data=csv_data, headers=headers)

print(response.json())