from evaluation.evaluation import Evaluation
from services.llm_service import LLMService
from services.command_line_service import get_cl_args_eval

def main():
    # Get command line arguments
    args = get_cl_args_eval()

    # Initialize the LLM service with API key and base URL
    llm_service = LLMService(llm_name=args.llm)

    evaluation = Evaluation(llm_service=llm_service, **vars(args))
    evaluation.run()

if __name__ == "__main__":
    main()