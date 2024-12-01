from food_db import get_db
from food_image_embedding import get_image_embedding, get_text_embedding, compute_similarity
import numpy as np


def perform_retrieval_with_image(db, image_path, k=15):
    img_feat = get_image_embedding(image_path)
    img_query = np.array(img_feat)  # Ensure it's a NumPy array
    return db.similarity_search_by_vector(img_query[0], k=k)


def perform_retrieval_with_text(db, text_query, k=2):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': k})
    return retriever.invoke(text_query)


if __name__ == "__main__":

    db = get_db()

    image_path = "../data/images/chicken_pasta.jpg"
        
    # try:
    #     image_results = perform_retrieval_with_image(db, image_path)
    #     print("Image Query Results:")
    #     for result in image_results:
    #         print(result)
    # except Exception as e:
    #     print(f"Error in image query: {e}")

    text_query = (
        # "Penne pasta"
        # "Cooked chicken pieces"
        # "Broccoli florets"
        # "Creamy sauce (likely made with cream, milk, or cheese)"
        "Olive oil or butter"
    )
    try:
        text_results = perform_retrieval_with_text(db, text_query)
        print("\nText Query Results:")
        for result in text_results:
            print(result)
    except Exception as e:
        print(f"Error in text query: {e}")
