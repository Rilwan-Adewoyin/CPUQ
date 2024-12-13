# This is the repository for the CPUQ research project: Categorical Prompt Uncertainty Quantification

## Setup
- Git Clone this repository into a folder called CPUQ
- Create conda/pyenv environment using the requirements.txt file
- Follow instructions in the Data section below to download the exact dataset used 

## How To Use
- Examples of how to key functions can be found in the example_scripts folder

## Data 
### Data Download
- Downaload the file at the following link: https://drive.google.com/file/d/14sIiCiT8ZPtvEI1DG8rGiMfZznxlYrt_/view?usp=sharing
- It is a tar.gz file so extract it into the repository and merge it with the /data folder in the repository

## Experiments

#### Probabilistic Prediction for Whether A Budget Item Affects an Socio-Economic Indicator
- bash example_scripts/predict_budgetitem_to_indicator.sh GPU_IDS
- GPU_IDS is a comma separated list of GPU IDs to use for the prediction

#### Probabilistic Prediction for Whether A Socio-Economic Indicator Affects Another Socio-Economic Indicator
- bash example_scripts/predict_indicator_to_indicator.sh GPU_IDS
- GPU_IDS is a comma separated list of GPU IDs to use for the prediction

## Analysis of results
- See 