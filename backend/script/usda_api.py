import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEY = os.getenv('USDA_API_KEY')

def search_food(query):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={query}&api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

if __name__ == "__main__":
    query = "apple"
    result = search_food(query)
    first_food = result["foods"][0]
    first_food_json = json.dumps(first_food, indent=4)  # Convert to JSON with indentation for readability
    print(first_food_json)
    # print(result)