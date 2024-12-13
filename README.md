# CPUQ: Categorical Prompt Uncertainty Quantification

## Overview
This repository contains the implementation of the CPUQ research project, focusing on Categorical Prompt Uncertainty Quantification.

## Setup
1. Clone the repository:   ```bash
   git clone <repository-url> CPUQ   ```
2. Create a conda/pyenv environment using the provided requirements:   ```bash
   conda create -n cpuq -f requirements.txt   ```
3. Download and prepare the dataset (see [Data Section](#data))

## Usage
Example scripts demonstrating key functionalities can be found in the `example_scripts/` directory.

## Data
### Dataset Download Instructions
1. Download the dataset from [Google Drive](https://drive.google.com/file/d/14sIiCiT8ZPtvEI1DG8rGiMfZznxlYrt_/view?usp=sharing)
2. Extract the downloaded `tar.gz` file
3. Merge the extracted contents with the `/data` folder in the repository

## Experiments

### Probabilistic Budget Impact Prediction
Predict whether a budget item affects a socio-economic indicator:

bash example_scripts/predict_budgetitem_to_indicator.sh <GPU_IDS>


### Probabilistic Indicator Relationship Prediction
Predict whether a socio-economic indicator affects another indicator:

bash example_scripts/predict_indicator_to_indicator.sh <GPU_IDS>

Note: `GPU_IDS` should be a comma-separated list of GPU IDs to use for prediction.

## Results Analysis
Detailed analysis of results can be found in:
- `prompt_engineering/analysis/analyses.ipynb`
