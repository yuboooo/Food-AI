import os
from dotenv import load_dotenv
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

file_path = "../data/food_db/food_descriptions.csv"
persist_directory = "../data/food_db/chroma_food_db"


openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
def get_db():
    if not os.path.exists(persist_directory):
        data = pd.read_csv(file_path)
        text_data = data['Description'].tolist()
        db = Chroma.from_texts(text_data, openai_embeddings, persist_directory=persist_directory)
    else:
        db = Chroma(persist_directory=persist_directory, embedding_function=openai_embeddings)
    return db
