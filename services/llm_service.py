import openai
import requests
from openai.types.chat import ChatCompletionUserMessageParam


class LLMService:
    """Service for interacting with a Language Model (LLM) via OpenAI API."""

    def __init__(self, api_key: str, base_url: str):
        """
        Initializes the LLMService with API key and base URL.
        Args:
            api_key (str): The API key for the LLMs.
            base_url (str): The base URL for the LLM access.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def get_llm_response(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generates a response from the LLM based on the provided prompt and parameters.
        Args:
            prompt (str): The input prompt for the LLM.
            model (str): The model to use for generating the response.
            temperature (float): Controls the randomness of the output.
            max_tokens (int): Maximum number of tokens to generate in the response.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_available_models(self) -> None:
        """Fetches and prints the list of available models from the LLM service."""
        url = self.base_url + "/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            print("Available models:")
            for model in models_data:
                print("-", model["id"])
        else:
            print("Error:", response.status_code, response.text)