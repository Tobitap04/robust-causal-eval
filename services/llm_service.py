import openai
import requests
from openai.types.chat import ChatCompletionUserMessageParam


class LLMService:
    """Service for interacting with a Language Model (LLM) via OpenAI API."""

    def __init__(self, api_key: str, base_url: str, llm_name: str) -> None:
        """
        Initializes the LLMService with API key and base URL.
        Args:
            api_key (str): The API key for the LLMs.
            base_url (str): The base URL for the LLM access.
            llm_name (str): The name of the LLM to use.
        """
        self.api_key = api_key
        self.base_url = base_url
        # Check if the provided LLM name is available
        if llm_name not in self.get_available_models():
            raise ValueError(f"LLM model '{llm_name}' is not available. Available models: {self.get_available_models()}. Please check the model name.")
        self.llm_name = llm_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def get_llm_response(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str | None:
        """
        Generates a response from the LLM based on the provided prompt and parameters.
        Args:
            prompt (str): The input prompt for the LLM.
            temperature (float): Controls the randomness of the output.
            max_tokens (int): Maximum number of tokens to generate in the response.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            # TODO: Add support for system message if needed
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            params = {
                "model": self.llm_name,
                "messages": messages,
            }
            if max_tokens is not None:
                params["max_tokens"] = str(max_tokens)
            if temperature is not None:
                params["temperature"] = str(temperature)

            response = self.client.chat.completions.create(**params)


            # Check if the response contains a <think> tag and handle it accordingly
            if "</think>" in response.choices[0].message.content:
                return response.choices[0].message.content.split("</think>", 1)[-1].strip()
            else:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_available_models(self) -> list | None:
        """
        Fetches list of available models from the LLM service.
        Returns:
            list: A list of available model IDs.
        """
        url = self.base_url + "/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            return [model["id"] for model in models_data]
        else:
            print("Error:", response.status_code, response.text)
            return None