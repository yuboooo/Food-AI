from openai import OpenAI

class QuerySimplifier:
    def __init__(self, api_key: str):
        """
        Initialize the QuerySimplifier class with OpenAI API key.
        Args:
            api_key (str): OpenAI API key.
        """
        self.client = OpenAI(api_key=api_key)
        if not api_key:
            raise ValueError("API key must be provided.")

    def simplify_query(self, complex_query: str) -> str:
        """
        Simplify a complex query to make it more suitable for the USDA API.
        Args:
            complex_query (str): The initial, complex query string.
        Returns:
            str: Simplified query string.
        """
        prompt = (
            f"The following query is too complex for the USDA API: '{complex_query}'. "
            "Simplify this query slightly, keeping the meaning intact and ensuring it is suitable for obtaining "
            "nutritional data. Do not oversimplify or remove key elements that are essential for accurate results. "
            "Output only the adjusted query."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error during API call for query simplification: {str(e)}")
