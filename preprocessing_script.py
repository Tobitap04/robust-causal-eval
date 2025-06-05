from preprocessing.preprocessing import Preprocessing
from services.command_line_service import get_cl_args_preproc
from services.llm_service import LLMService


def main():
    # Get command line arguments
    args = get_cl_args_preproc()

    # Initialize the LLM service with API key and base URL
    llm_service = LLMService(llm_name=args.llm)

    preprocessing = Preprocessing(llm_service)
    preprocessing.run(args.nq)

if __name__ == "__main__":
    main()