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
You will be prompted to enter your LLM_API_KEY and LLM_API_URL.

## Data Preprocessing Instructions
The repository already contains a carefully preprocessed dataset `data/final.csv`, but if you want to run the preprocessing yourself, follow these steps. 

### Step 1: Download dataset
Before running any preprocessing scripts, you need to download the [Webis-CausalQA-22](https://webis.de/data/webis-causalqa-22.html) dataset and put the csv files of the original or random split into the `data/raw` directory. If you want to recreate our results, you have to exclude the `eli5` dataset.

### Step 2: Create sample
As a first step, you have to create a sample from the original dataset. This can be done by running the following command (this may take a few minutes):
```bash
python preprocessing_script.py create_sample --output_path data/sample.csv --target_size 10000
```
After this you can delete the `data/raw` directory, as it is no longer needed.