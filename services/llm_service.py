from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai.types.chat import ChatCompletionUserMessageParam
from ratelimit import limits, RateLimitException
from dotenv import load_dotenv
import requests
import logging
import openai
import time
import os


class LLMService:
    """Service for interacting with a Language Model (LLM) via OpenAI API."""
    RPM = 10  # Requests per minute

    def __init__(self, llm_name: str) -> None:
        """
        Initializes the LLMService with API key and base URL.
        Args:
            llm_name (str): The name of the LLM to use.
        """
        # Fetch API key and base url from environment variables
        load_dotenv("config.env")
        self.api_key = os.getenv("LLM_API_KEY")
        if self.api_key is None:
            raise RuntimeError("LLM_API_KEY is not set")
        self.base_url = os.getenv("LLM_BASE_URL")
        if self.base_url is None:
            raise RuntimeError("LLM_BASE_URL is not set")

        # Check if the provided LLM name is available
        if llm_name not in self.get_available_models():
            raise ValueError(f"LLM model '{llm_name}' is not available. Available models: {self.get_available_models()}. Please check the model name.")
        self.llm_name = llm_name
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @staticmethod
    def before_sleep(retry_state):
       """Logs the exception and waits before retrying.
       Args:
            retry_state (RetryCallState): The state of the retry call.
       """
       exc = retry_state.outcome.exception()
       if isinstance(exc, RateLimitException):
           time.sleep(getattr(exc, "period_remaining", 0))



    @retry(
       wait=wait_exponential(multiplier=1, min=2, max=30),
       stop=stop_after_attempt(7),
       retry=retry_if_exception_type((RateLimitException, requests.exceptions.RequestException, openai.RateLimitError)),
       before_sleep=before_sleep,
    )
    @limits(calls=RPM, period=60)
    def get_llm_response(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """
        Fetches a response from the LLM based on the provided prompt.
        Args:
        prompt (str): The input prompt for the LLM.
            temperature (float, optional): Sampling temperature for the response. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
        """
        try:
            messages: list[openai.types.chat.ChatCompletionUserMessageParam] = [
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

            content = response.choices[0].message.content

            # Remove the <think> tag of reasoning if present
            if "</think>" in content:
                return content.split("</think>", 1)[-1].strip()
            else:
                return content.strip()
        except Exception as e:
            logging.error(f"Error fetching LLM response: {e}")
            raise e

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