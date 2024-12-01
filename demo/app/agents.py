import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = st.secrets["general"]["OPENAI_API_KEY"]

def agent1_food_image_caption(encoded_image: str) -> str:
    """
    Take the food image (base64 encoded) and prompt (which ask to describe the food component in the image) and return the caption.
    """
    # Step 1: Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Step 2: Prompt
    prompt = "Identify the main food item in the image and list its major components or ingredients. Return the result as a plain, comma-separated string (e.g., Salmon (raw), White rice, Pineapple, Cucumber, Seaweed (wakame), Sesame seeds). Do not include brackets, quotes, or any other formatting."

    # Step 3: Return the caption
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=100
        )

        ingredients_str = response.choices[0].message.content.strip()
        ingredients = [item.strip() for item in ingredients_str.split(',')]
        return ingredients
    except Exception as e:
        raise Exception(f"Error during API call: {str(e)}")


def agent2_nutrition_augmentation(encoded_image: str, nutrition_info: dict) -> str:
    """
    Take the nutrition information and augment it with additional details.
    """
    # Step 1: Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    # Step 2: Prompt


    prompt = "The above nutrition facts, it describe the ingredient's nutrition per 100g. Can you then estimate the total nutrition info for the food in the provided image, based on the nutrition facts i provided to you, and also your own knowledge from your database, if you identified this food from your database, you can also directly use the information their. Simply return the nutrition info in a nice readable str format, make it concise, and easy to read. If you need to use scratch pad, you can use the scratch pad below to do your calculation. But keep the output clean(dont explicitly show our provided nutrition facts in response), especially provide a Summary Section in the end"

    prompt = f"{nutrition_info} \n\n{prompt}"
    


    # prompt = f"""You are a nutrition expert analyzing the dish in the provided image.

    #         AVAILABLE NUTRITION DATA (per 100g):
    #         {nutrition_info}

    #         ANALYSIS INSTRUCTIONS:
    #         1. Identify all visible ingredients in the dish
    #         2. Use provided nutrition facts and your knowledge to calculate total nutrition
    #         3. Show ingredient-by-ingredient breakdown
    #         4. Calculate and show the total values
    #         5. Include a brief summary and any relevant assumptions

    #         PROVIDE YOUR RESPONSE IN THIS FORMAT:

    #         Based on the ingredients visible in the image and their corresponding nutrition facts, here's an estimated nutritional breakdown:

    #         Ingredient-by-Ingredient Breakdown (per 100g):
    #         [Ingredient 1 Name]:
    #         • Energy: X kcal
    #         • Protein: X g
    #         • Total lipid (fat): X g

    #         [Ingredient 2 Name]:
    #         • Energy: X kcal
    #         • Protein: X g
    #         • Total lipid (fat): X g
    #         [Continue for all ingredients...]

    #         Total Nutrition Calculation:
    #         Energy: [Show addition of all ingredients] = X kcal
    #         Protein: [Show addition of all ingredients] = X g
    #         Total lipid (fat): [Show addition of all ingredients] = X g

    #         Final Summary:
    #         • Total Energy: X kcal
    #         • Total Protein: X g
    #         • Total Fat: X g

    #         Assumptions and Notes:
    #         - Include key assumptions about portions
    #         - Note if standard database values were used
    #         - Mention any significant adjustments made

    #         REQUIREMENTS:
    #         - Show calculations clearly but keep them organized
    #         - Round numbers to 2 decimal places
    #         - Use consistent units throughout
    #         - If using database values, clearly indicate this
    #         - Consider both visible evidence and typical serving sizes"""

    # Step 3: Return the augmented nutrition information
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=1000
        )

        augmented_nutrition_info = response.choices[0].message.content.strip()
        return augmented_nutrition_info
    except Exception as e:
        raise Exception(f"Error during API call: {str(e)}")