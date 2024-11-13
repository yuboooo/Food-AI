import os
from dotenv import load_dotenv
from pathlib import Path
from agent1_process_food_image import ImageToQuery  # Assuming ImageToQuery is in a file named image_to_query.py
from usda_api import search_food, save_results_to_file  # Assuming USDA functions are in a file named usda_query.py
from agent2_query_formation import QuerySimplifier  # Assuming QuerySimplifier is moved to query_simplifier.py

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Initialize the ImageToQuery and QuerySimplifier classes
    api_key = os.getenv('OPENAI_API_KEY')
    usda_api_key = os.getenv('USDA_API_KEY')

    if not api_key or not usda_api_key:
        raise ValueError("Both OPENAI_API_KEY and USDA_API_KEY must be provided in the environment.")

    analyzer = ImageToQuery(api_key)
    simplifier = QuerySimplifier(api_key)

    # Example image and prompt
    image_path = "../data/images/burger.jpg"
    prompt = (
        "Analyze the provided food image and generate a concise query string that describes the food in a way "
        "that is suitable for obtaining nutritional data from the USDA API. The query should focus on accurately "
        "identifying the food items visible in the image, such as 'cheeseburger with lettuce and tomato', and "
        "should avoid unnecessary details or unrelated information."
    )

    try:
        # Generate query from image
        query = analyzer.generate_query(image_path, prompt)
        print(f"Generated Query: {query}")

        # Perform USDA API search with the generated query
        for attempt in range(5):
            results = search_food(query, page_size=5)

            if results.get("totalHits", 0) > 0:
                break  # Exit the loop if results are found

            print("No results found. Attempting to simplify the query...")
            query = simplifier.simplify_query(query)
            print(f"Simplified Query (Attempt {attempt + 1}): {query}")
        else:
            print("No results found after 5 attempts.")

        # Save the results to a JSON file
        save_results_to_file(query, results)

    except Exception as e:
        print(f"Error: {str(e)}")