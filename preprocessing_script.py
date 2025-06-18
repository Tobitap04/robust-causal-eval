from preprocessing.preprocessing import Preprocessing
from services.command_line_service import get_cl_args_preproc
from services.llm_service import LLMService


def main():
    # Get command line arguments
    args = get_cl_args_preproc()




    if args.function == "create_sample":
        if not args.target_size or not args.output_path:
            raise ValueError("Both target_size and output_path must be specified for create_sample function.")
        preprocessing = Preprocessing()
        preprocessing.create_sample(target_size=args.target_size, output_path=args.output_path)
    elif args['function'] == "filter_questions":
        pass
        # Initialize the LLM service with API key and base URL
        # llm_service = LLMService(llm_name=args.llm)
        # preprocessing.filter_questions()
    elif args['function'] == "sample_lookup":
        pass
        # preprocessing.sample_lookup()



if __name__ == "__main__":
    main()