import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from ratelimit import limits, RateLimitException
from tenacity import retry, wait_exponential, stop_after_attempt


class LLMService:
    """Service for interacting with a Language Model (LLM) via OpenAI API."""
    RPM = 10  # Requests per minute

    def __init__(self, llm_name: str) -> None:
        """
        Initializes the LLMService with API key and base URL.
        Args:
            llm_name (str): The name of the LLM to use.
        Raises:
            ValueError: If environment variables are not set or if the LLM name is not available.
        """
        # Fetch API key and base url from environment variables
        load_dotenv("config.env")
        api_key = os.getenv("LLM_API_KEY")
        if api_key is None:
            raise ValueError("LLM_API_KEY is not set")
        base_url = os.getenv("LLM_BASE_URL")
        if base_url is None:
            raise ValueError("LLM_BASE_URL is not set")
        self.llm_name = llm_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90
        )

    @staticmethod
    def handle_rate_limit(retry_state):
        """Handles rate limit exceptions by sleeping for the remaining period.
        Args:
             retry_state (RetryCallState): The state of the retry call.
        """
        exc = retry_state.outcome.exception()
        if isinstance(exc, RateLimitException):
            time.sleep(getattr(exc, "period_remaining", 0))

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(7),
        before_sleep=handle_rate_limit,
    )
    @limits(calls=RPM, period=60)
    def get_llm_response(self, prompt: str, temperature: float = 1) -> str:
        """
        Fetches a response from the LLM based on the provided prompt.
        Should always be covered in try-except block to handle exceptions.
        Args:
            prompt (str): The input prompt for the LLM.
            temperature (float, optional): Sampling temperature for the response. Defaults to None.
        Returns:
            str: The response from the LLM, with any </think> tags removed.
        Raises:
            RetryError: If the request fails after retries.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                **({"extra_body": {"cache": {"no-cache": True}}} if "gwdg" in self.llm_name else {})
                # No caching for gwdg models
            )
            response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    response += delta

            # Remove the </think> tag of reasoning if present
            if "</think>" in response:
                return response.split("</think>", 1)[-1].strip()
            else:
                return response.strip()

        except Exception as e:
            print()
            logging.warning(f"LLMService: Error fetching LLM response: {e}")
            raise e
