import requests
from dotenv import load_dotenv
import os
import json

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv('USDA_API_KEY')

# Function to search for food using the USDA API
def search_food(query, data_type=None, page_size=10):
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": query,
        "api_key": API_KEY,
        "pageSize": page_size,
    }
    if data_type:
        params["dataType"] = data_type

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

# Function to save results to a file
def save_results_to_file(query, results):
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(output_dir, f"{query.replace(' ', '_')}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results for query '{query}' saved to {file_path}")

if __name__ == "__main__":
    # Test cases with increasing specificity
    test_queries = [
        {"query": "apple"},  # General query
        {"query": "apple raw"},  # More specific
        {"query": "apple raw with skin", "data_type": ["Foundation"]},  # Highly specific
        {"query": "grilled chicken breast", "data_type": ["Foundation"], "page_size": 5},  # Complex description
        {"query": "chicken cooked", "data_type": ["SR Legacy"], "page_size": 5},  # Cooked food
        {"query": "banana"},  # Another general query
        {"query": "banana raw", "data_type": ["Foundation"], "page_size": 5},  # Narrowing down
    ]

    # Run tests and save results
    for test in test_queries:
        query = test.get("query")
        data_type = test.get("data_type")
        page_size = test.get("page_size", 5)

        try:
            print(f"Running query: {query}")
            results = search_food(query, data_type=data_type, page_size=page_size)
            save_results_to_file(query, results)
        except Exception as e:
            print(f"Error with query '{query}': {e}")
