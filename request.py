import requests

url = "http://127.0.0.1:5000/analyze"
data = {"year": "2017", "quarter": "Q1"}

response = requests.post(url, data=data)

if response.status_code == 200:
    with open("analyze_results.csv", "wb") as file:
        file.write(response.content)
    print("CSV file saved as analyze_results.csv")
else:
    print(f"Error: {response.status_code} - {response.text}")
