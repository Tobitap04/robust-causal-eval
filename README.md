# robust-causal-eval
## Installation


At first, you need to clone the repository:
```
git clone https://github.com/Tobitap04/robust-causal-eval.git
cd robust-causal-eval
```

We recommend using `Python 3.10`, as this is the version the project was developed and tested with.  
Optionally create a python venv or conda environment. 

With `Anaconda3`:
```
conda create --name robust-causal-eval python=3.10 
conda activate robust-causal-eval
```

With `python`:
```
python -m venv robust-causal-eval
source robust-causal-eval/bin/activate
```

To install all dependencies and create the required `config.env` file, run:
```bash
python setup.py
```
You will be prompted to enter your LLM_API_KEY and LLM_BASE_URL.

> **Note:** Since preprocessing and evaluation steps can take a long time, we recommend running the project inside a `tmux` session to avoid interruptions.

## Data Preprocessing Instructions
The repository already contains a carefully preprocessed and filtered sample of the [Webis-CausalQA-22](https://webis.de/data/webis-causalqa-22.html) dataset at `data/final_sample.csv`. If you are interested in how this version was created, you can follow the steps below.  
Otherwise, you can skip directly to the Evaluation section. When creating your own sample, ensure to not overwrite existing files in the `data` directory.
### Step 1: Download dataset
First, download the [Webis-CausalQA-22](https://webis.de/data/webis-causalqa-22.html) dataset and copy all CSV files from `Webis-CausalQA-22-v-2.0/input/original-splits` into the `data/raw` directory.  
Next, run the following script. It will download the Eli5 dataset, remove columns that are not needed for this project, and merge the training and validation splits. This process may take a few minutes:
```bash
python preprocessing_script.py data_setup
```
### Step 2: Create sample
Now you need to create a sample from the dataset. We decided to start with a sample size of 6,000 and exclude the following datasets: searchqa, triviaqa, newsqa, paq and hotpotqa. For the rationale behind this decision, please refer to our paper.  
To generate the sample called `unfiltered_sample_new.csv` stored in the `data` directory, run the following script:
```bash
python preprocessing_script.py create_sample --output_path data/unfiltered_sample_new.csv --exclude searchqa triviaqa newsqa hotpotqa paq --nq 6000
```
(After this you can delete the `data/raw` directory, as it is no longer needed.)

### Step 3: Filter sample
To filter the sample, we removed all questions-answer pairs that did not meet the criteria defined in our paper. For this purpose, we developed a series of filtering functions tailored to create a high-quality dataset for our evaluation.  
The filters were applied sequentially (in the order specified below), and the intermediate results were saved after each step.  
Filtering is performed using a large language model (LLM) with a rate limit set to 10 requests per minute (this can be changed in `services/llm_service.py`). As a result, filtering larger samples may take **several hours**.  
Each filter uses a few-shot prompting strategy, defined in `preprocessing/filter_funcs.py`.  
To run the filtering, use the following command with the appropriate filter name:
```bash
python preprocessing_script.py filter_questions --filter causal_chain --input_path data/unfiltered_sample_new.csv --output_path data/filtered_01_causal_chain_new.csv
```
The available filters are:
- `causal_chain`: Removes all question-answer pairs that do not clearly exhibit a causal chain between the question and the answer.
- `answer`: Removes all question-answer pairs where the answer is not relevant, accurate, or appropriately formatted.
- `question`: Removes all question-answer pairs where the question is not clearly stated or contains contextual ambiguities.

#### Inspection Tools
To refine our filtering prompts, we used this command to evaluate the dataset after each step and identify common issues. It displays `nq` random entries from the specified file:
```bash
python preprocessing_script.py sample_lookup --input_path data/unfiltered_sample_new.csv --nq 100
```
You can also use the following function to check the number of questions per dataset contained in the given path:
```bash
python preprocessing_script.py sample_stats --input_path data/unfiltered_sample_new.csv
```
### Step 4: Create perturbations
Finally, perturbations are created for the filtered sample. To ensure high-quality and diverse modifications, we use a large language model (LLM), which outperformed other methods in our tests.  
All perturbations are generated using a one-shot prompting strategy defined in `preprocessing/perturbation_funcs.py`. The only exception is the character-level variant, which is generated using the typo library instead of the LLM.
You can optionally set the perturbation intensity using `--intensity` (25, 50, 75, or 100). However, we recommend leaving it unset, as each type has a predefined default based on prior evaluation.  
To create the perturbations, run the following command (this may also take **several hours**):
```bash
python preprocessing_script.py create_perturbs --input_path data/filtered_03_question_new.csv --output_path data/final_sample_new.csv
```

> **Note:** Per default, we use the `gwdg.qwen2.5-72b-instruct` model for filtering and perturbation creation. You can specify a different LLM by using the `--llm` parameter.  

## LLM Evaluation Instructions


- Help cl function für beides scripts erwähnen
- extra parameter erklären