# robust-causal-eval
## Installation


At first, you need to clone the repository:
```
git clone https://github.com/Tobitap04/robust-causal-eval.git
cd robust-causal-eval
```
Optionally create a python venv or conda environment. Requires `Python >= 3.10`.

With `Anaconda3`:
```
conda create --name robust-causal-eval python=3.10 
conda init bash
source ~/.bashrc
conda activate robust-causal-eval
```

With `python`:
```
python -m venv robust-causal-eval
source robust-causal-eval/bin/activate
```

We recommend using a `tmux` session for this project, as filtering the dataset and evaluating the models can take a significant amount of time.
You can start a new session with the first command. To detach from the session, press `Ctrl + b`, then `d`. You can later reattach to the session with the second command without losing your progress:
```
tmux new -s robust-causal-eval
tmux attach -t robust-causal-eval
```

To install all dependencies and create the required `config.env` file, run:
```bash
python setup.py
```
You will be prompted to enter your LLM_API_KEY and LLM_BASE_URL.

## Data Preprocessing Instructions
The repository already contains a carefully preprocessed and filtered sample of the [Webis-CausalQA-22](https://webis.de/data/webis-causalqa-22.html) dataset at `data/final.csv`. If you are interested in how this version was created, you can follow the steps below.  
Otherwise, you can skip directly to the Evaluation section.

### Step 1: Download dataset
First, download the [Webis-CausalQA-22](https://webis.de/data/webis-causalqa-22.html) dataset and copy all CSV files from `Webis-CausalQA-22-v-2.0/input/original-splits` into the `data/raw` directory.  
Next, run the following script. It will download the Eli5 dataset, remove columns that are not needed for this project, and merge the training and validation splits. This process may take a few minutes:
```bash
python preprocessing_script.py data_setup
```
### Step 2: Create sample
Now you need to create a sample from the dataset. We decided to start with a sample size of 10,000 and exclude the following datasets: searchqa, triviaqa, newsqa, and hotpotqa. For the rationale behind this decision, please refer to our paper.  
To generate the sample called `sample.csv` stored in the `data` directory, run the following script (this may take a few minutes):
```bash
python preprocessing_script.py create_sample --output_path data/sample.csv --exclude searchqa triviaqa newsqa hotpotqa --nq 10000
```
(After this you can delete the `data/raw` directory, as it is no longer needed.)

### Step 3: Filter sample
To filter the sample, we removed all questions that did not meet the criteria defined in our paper. For this purpose, we developed a series of filtering functions tailored to create a high-quality dataset for our evaluation.  
The filters were applied sequentially (in the order specified below), and the intermediate results were saved after each step. The final filtered sample is available at `data/final.csv`.  
Filtering is performed using a large language model (LLM) with a rate limit set to 10 requests per minute (this can be changed in `services/llm_service.py`). As a result, filtering larger samples may take **several hours**.  
Each filter uses a few-shot prompting strategy, defined in `preprocessing/prompt_builder.py`.  
To run the filtering, use the following command with the appropriate filter name:
```bash
python preprocessing_script.py filter_questions --filter causal_chain --input_path data/sample.csv --output_path data/filtered_causal_chain.csv
```
The available filters are:
- `causal_chain`: This filter removes all question and answer pairs that do not exhibit a causal chain.
- `filter_2`: ...
- `filter_3`: ...
- `filter_4`: ...

To refine our filtering prompts, we used this command to evaluate the dataset after each step and identify common issues. It displays `nq` random entries from the specified file:
```bash
python preprocessing_script.py sample_lookup --input_path data/sample.csv --nq 100
```
You can also use the following function to check the number of questions per dataset contained in the given path:
```bash
python preprocessing_script.py sample_stats --input_path data/sample.csv
```
## LLM Evaluation Instructions

- Python version: 3.10
- Wie viel SPeicher am ende bendötigt? (venv datei checken)
- Help function für beides scripts erwähnen