from preprocess import encode_image
from agents import agent1_food_image_caption, agent2_nutrition_augmentation
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import boto3
from PIL import Image
import os


# def get_db_json():
#     return Chroma(
#         collection_name="food_items_collection",
#         embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
#         persist_directory="../data/food_db/vector_db_json"
#     )

def download_s3_bucket(bucket_name, local_dir):
    # Create an S3 client
    s3 = boto3.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name}
    
    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                local_file_path = os.path.join(local_dir, key)

                # Create local directory structure if it doesn't exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                print(f"Downloading {key} to {local_file_path}")
                s3.download_file(bucket_name, key, local_file_path)

# Function to load Chroma database
def get_db_json():
    # Define S3 and local paths
    bucket_name = "food-ai-db" 
    local_dir = "../data/food_db_cloud/" 

    # Call the function
    download_s3_bucket(bucket_name, local_dir)

    db_path = os.path.join(local_dir, "vector_db_json")

    # Load the Chroma database
    return Chroma(
        collection_name="food_items_collection",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=db_path
    )

if __name__ == "__main__":

    # Streamlit app
    st.title("Food Nutrition Analyzer")

    st.header("Upload a Food Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Food Image", use_container_width=True)

        # Encode image and extract ingredients
        st.write("Processing image to extract food ingredients...")
        encoded_image = encode_image(uploaded_file)
        ingredients = agent1_food_image_caption(encoded_image)
        st.subheader("Extracted Food Ingredients")
        st.write(ingredients)

        # Get the database
        db = get_db_json()

        # Retrieve nutrition info for each ingredient
        st.write("Fetching nutrition information for ingredients...")
        nutrition_info = {}
        for ingredient in ingredients:
            similar_doc = db.similarity_search(ingredient, k=1)
            food_description = similar_doc[0].page_content if similar_doc else None
            metadata = similar_doc[0].metadata
            nutrition_info[food_description] = metadata

        st.subheader("Nutrition Facts for Each Ingredient (per 100g)")
        for description, metadata in nutrition_info.items():
            st.write(f"**{description}**: {metadata}")

        # Augmented nutrition data
        st.write("Generating augmented nutrition information...")
        nutrition_augmentation = agent2_nutrition_augmentation(encoded_image, nutrition_info)
        st.subheader("Augmented Nutrition Information")
        st.write(nutrition_augmentation)
