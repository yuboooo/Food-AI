import os
import base64
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ImageToQuery:
    def __init__(self, api_key: str):
        """
        Initialize the ImageToQuery class with OpenAI API key.
        Args:
            api_key (str): OpenAI API key.
        """
        self.client = OpenAI(api_key=api_key)
        if not api_key:
            raise ValueError("API key must be provided.")

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        Args:
            image_path (str): Path to image file.
        Returns:
            str: Base64 encoded image.
        """
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")

    def generate_query(self, image_path: str, prompt: str) -> str:
        """
        Generate a good query for the USDA API from an image and a prompt.
        Args:
            image_path (str): Path to image file.
            prompt (str): Instruction/prompt for analysis.
        Returns:
            str: Generated query string.
        """
        base64_image = self.encode_image(image_path)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=100
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error during API call: {str(e)}")

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    analyzer = ImageToQuery(api_key)

    image_path = "../data/images/burger.jpg"
    prompt = (
        "Describe the food in the image, make it concise and simple. Try to indicate all the food component in the image."
    )

    try:
        query = analyzer.generate_query(image_path, prompt)
        print(f"Generated Query: {query}")
    except Exception as e:
        print(f"Error: {str(e)}")
