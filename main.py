from dotenv import load_dotenv
from services.llm_service import LLMService
from data.preprocessing import Preprocessing
import os

def main():
    # Fetch API key and base url from environment variables
    load_dotenv("config.env")
    api_key = os.getenv("LLM_API_KEY")
    if api_key is None:
        print("LLM_API_KEY is not set")
        exit(1)
    base_url = os.getenv("LLM_BASE_URL")
    if base_url is None:
        print("LLM_BASE_URL is not set")
        exit(1)

    # Initialize the LLM service with API key and base URL
    llm_service = LLMService(api_key=api_key, base_url=base_url)

    # Starts the preprocessing
    preprocessing = Preprocessing(llm_service)
    preprocessing.run(15)

if __name__ == "__main__":
    main()