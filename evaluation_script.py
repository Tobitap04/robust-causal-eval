from dotenv import load_dotenv
from evaluation.evaluation import Evaluation
from services.llm_service import LLMService
from services.command_line_service import get_cl_args
from transformers import logging
import os

def main():
    # Get command line arguments
    args = get_cl_args()

    # Fetch API key and base url from environment variables
    load_dotenv("config.env")
    api_key = os.getenv("LLM_API_KEY")
    if api_key is None:
        raise RuntimeError("LLM_API_KEY is not set")
    base_url = os.getenv("LLM_BASE_URL")
    if base_url is None:
        raise RuntimeError("LLM_BASE_URL is not set")

    # Initialize the LLM service with API key and base URL
    llm_service = LLMService(api_key=api_key, base_url=base_url, llm_name=args.llm)
    '''
    # Print available models
    # print(llm_service.get_available_models())
    
    # Starts the preprocessing
    preprocessing = Preprocessing(llm_service)
    preprocessing.run(40)
    '''
    logging.set_verbosity_error()  # Suppress warnings from transformers library

    evaluation = Evaluation(llm_service=llm_service, **vars(args))
    evaluation.run()
if __name__ == "__main__":
    main()