from preprocessing.preprocessing import Preprocessing
from services.command_line_service import get_cl_args_preproc
from services.llm_service import LLMService
from preprocessing.data_setup import run_data_setup


def main():
    # Get command line arguments
    args = get_cl_args_preproc()

    if args.function == "filter_questions":
        if not args.input_path or not args.llm or not args.output_path or not args.filter:
            raise ValueError("input_path, output_path, llm and filter must be specified for filter_questions function.")
        llm_service = LLMService(llm_name=args.llm)
        preprocessing = Preprocessing(llm_service=llm_service)
        preprocessing.filter_questions(input_path=args.input_path, output_path=args.output_path, filter_type=args.filter)
    elif args.function == "create_sample":
        if not args.nq or not args.output_path:
            raise ValueError("Both nq and output_path must be specified for create_sample function.")
        Preprocessing.create_sample(nq=args.nq, output_path=args.output_path, exclude=args.exclude)
    elif args.function == "sample_lookup":
        if not args.input_path or not args.nq:
            raise ValueError("input_path and nq must be specified for sample_lookup function.")
        pass
        Preprocessing.sample_lookup(input_path=args.input_path, nq=args.nq)
    elif args.function == "data_setup":
        run_data_setup()
    elif args.function == "sample_stats":
        if not args.input_path:
            raise ValueError("input_path must be specified for sample_stats function.")
        Preprocessing.sample_stats(input_path=args.input_path)
    elif args.function == "create_perturbs":
        if not args.input_path or not args.llm or not args.output_path:
            raise ValueError("input_path, output_path and llm must be specified for create_perturbs function.")
        llm_service = LLMService(llm_name=args.llm)
        preprocessing = Preprocessing(llm_service=llm_service)
        preprocessing.create_perturbs(input_path=args.input_path, output_path=args.output_path, intensity=args.intensity)

if __name__ == "__main__":
    main()