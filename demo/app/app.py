from preprocess import encode_image
from agents import agent1_food_image_caption, agent2_nutrition_augmentation
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from PIL import Image


def get_db_json():
    return Chroma(
        collection_name="food_items_collection",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="../data/food_db/vector_db_json"
    )

if __name__ == "__main__":

    image_path = "../data/images/mcbigmac.jpg"
    # encoded_image = encode_image(image_path)


    # ingredients = agent1_food_image_caption(encoded_image)
    # print("Food ingredients:", ingredients)


    # db = get_db_json()


    # nutrition_info = {}
    # for ingredient in ingredients:
    #     print("Ingredient:", ingredient)
    #     similar_doc = db.similarity_search(ingredient, k=1)
    #     food_description = similar_doc[0].page_content if similar_doc else None
    #     metadata = similar_doc[0].metadata
    #     nutrition_info[food_description] = metadata

    # print(nutrition_info)

    # nutrition_augmentation = agent2_nutrition_augmentation(encoded_image, nutrition_info)
    # print(nutrition_augmentation)



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

        st.subheader("Nutrition Information for Each Ingredient")
        for description, metadata in nutrition_info.items():
            st.write(f"**{description}**: {metadata}")

        # Augmented nutrition data
        st.write("Generating augmented nutrition information...")
        nutrition_augmentation = agent2_nutrition_augmentation(encoded_image, nutrition_info)
        st.subheader("Augmented Nutrition Information")
        st.write(nutrition_augmentation)

        # # Display total nutrition info
        # st.subheader("Total Nutrition Information")
        # total_nutrition = {}

        # for metadata in nutrition_info.values():
        #     for key, value in metadata.items():
        #         try:
        #             # Attempt to convert the value to a float for summation
        #             numeric_value = float(value)
        #             total_nutrition[key] = total_nutrition.get(key, 0) + numeric_value
        #         except (ValueError, TypeError):
        #             # Handle non-numeric values (e.g., strings) gracefully
        #             total_nutrition[key] = value if key not in total_nutrition else total_nutrition[key]

        # st.write(total_nutrition)