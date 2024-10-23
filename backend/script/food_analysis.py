import os
import base64
from datetime import datetime
from openai import OpenAI
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ImageAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ImageAnalyzer with OpenAI API key.
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment.
        """
        self.output_dir = Path("../outputs")
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        if not api_key and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        Args:
            image_path (str): Path to image file
        Returns:
            str: Base64 encoded image
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")

    def analyze_image(self, image_path: str, prompt: str) -> Dict:
        """
        Analyze image with GPT-4 Vision based on prompt.
        Args:
            image_path (str): Path to image file
            prompt (str): Instruction/prompt for analysis
        Returns:
            Dict: Analysis result with metadata
        """
        # Verify image exists
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Encode image
        base64_image = self.encode_image(image_path)

        try:
            # Create message with image and prompt
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Note: Changed from gpt-4o-mini to gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Return structured result
            return {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "prompt": prompt,
                "analysis": response.choices[0].message.content
            }
        except Exception as e:
            raise Exception(f"Error during API call: {str(e)}")
        
    def export_to_markdown(self, results: List[Dict], filename: Optional[str] = None) -> str:
        """
        Export analysis results to a markdown file.
        Args:
            results: List of analysis results
            filename: Optional custom filename
        Returns:
            str: Path to the created markdown file
        """
        if not filename:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Image Analysis Results\n\n")
            for result in results:
                f.write(f"## Analysis for: {Path(result['image_path']).name}\n")
                f.write(f"- **Timestamp:** {result['timestamp']}\n")
                f.write(f"- **Prompt:** {result['prompt']}\n\n")
                f.write("### Analysis:\n")
                f.write(f"{result['analysis']}\n\n")
                f.write("---\n\n")
        
        return str(filepath)

def main():
    # Example usage
    analyzer = ImageAnalyzer()
    
    # Example image paths and prompts
    analyses = []
    image_prompts = [
    (
        "../data/images/burger.jpg",
        "Describe what's in this image in detail."
    ),
    (
        "../data/images/burger.jpg",
        """
        Analyze the provided image of a meal. Identify all visible food items and estimate their quantities.
        For each identified food item, determine and report the following nutritional details:
        - Calories
        - Macronutrients (carbohydrates, proteins, fats)
        - Micronutrients (vitamins and minerals, if possible)
        
        Additionally, provide an estimation of the total calorie count for the entire meal.
        Describe the serving sizes in standard units (e.g., cups, tablespoons, grams) to help quantify each food item.
        If possible, note any visible condiments or toppings that could affect the nutritional value.
        The goal is to provide a detailed nutritional breakdown that can assist users in tracking their daily dietary intake against specific nutritional and calorie goals.
        """
    )
]

    
    try:
        # Analyze images
        for image_path, prompt in image_prompts:
            result = analyzer.analyze_image(image_path, prompt)
            analyses.append(result)
        
        # Export results to markdown
        md_path = analyzer.export_to_markdown(analyses, "burger_analysis.md")
        
        print(f"Results exported to:\nMarkdown: {md_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()