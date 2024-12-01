import open_clip
import numpy as np
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image

# Model configuration
MODEL_NAME = "ViT-B-32"
CHECKPOINT = "laion2b_s34b_b79k"

clip_embd = OpenCLIPEmbeddings(model_name=MODEL_NAME, checkpoint=CHECKPOINT)

def get_image_embedding(image_path):
    return np.array(clip_embd.embed_image([image_path]))

def get_text_embedding(texts):
    return np.array(clip_embd.embed_documents(texts))

def compute_similarity(embedding1, embedding2):
    return np.matmul(embedding1, embedding2.T)
