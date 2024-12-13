# Collect the information for the language model to use
import os
from argparse import ArgumentParser
from prompt_engineering import predict 
import yaml
from agent_based_modelling.i2i_edge_estimation.add_i2i_edge_weights import entropy_edge_weights
import pandas as pd
from scipy.stats import entropy

parser = ArgumentParser()
# parser.add_argument('--exp_idx', type=int, choices=[0,1,2], default=0)
# parser.add_argument('--experiment_dir', type=str, default=os.path.join('prompt_engineering','output','spot','exp_i2i_30b_distr','exp_upllama30b_non_uc') )

parser.add_argument('--cpuq_verbalize', type=str, choices=['cpuq','verbalize'] )
parser.add_argument('--debugging', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--finetuned', action='store_true', default=False)
parser.add_argument('--finetune_dir', type=str, default='prompt_engineering/finetune/ckpt')
parser.add_argument('--input_file', type=str, default='./data/spot/spot_b2i_networks/b2i_candidates.csv', help='Path to the file containing the input data' )

parser.add_argument('--exp_group', type=str, default='ppi_b2i_7bn' )
parser.add_argument('--exp_name', type=str, default='sbeluga7b' )
parser.add_argument('--finetune_version', type=int, default=4)
parser.add_argument('--llm_name', type=str, default='stabilityai/StableBeluga-7B')
parser.add_argument('--local_or_remote', type=str, default='local')
parser.add_argument('--unbias_categorisations', action='store_true', default=False)
parser.add_argument('--sampling_method', type=str, default=None)
parser.add_argument('--sample_count', type=int, default=1)
parser.add_argument('--use_system_prompt', type=lambda val: float(int(val)), default=True)
parser.add_argument('--override_double_quant', action='store_true', default=False)

parse_kwargs = parser.parse_args()

# Defining kwargs from prompting strategy
# i2i_model_kwargs = yaml.safe_load( open( os.path.join( parse_kwargs.experiment_dir, 'config.yaml'), "r" ) )

if parse_kwargs.cpuq_verbalize == 'cpuq':
    prompt_style = 'categorise'
    parse_style = 'categories_perplexity'
    # model_kwargs['exp_group'] = model_kwargs['exp_group'] + '_cpuq'

elif parse_kwargs.cpuq_verbalize == 'verbalize':
    prompt_style = 'yes_no'
    parse_style = 'rules'
    # model_kwargs['exp_group'] = model_kwargs['exp_group'] + '_verbalize'


delattr(parse_kwargs, 'cpuq_verbalize')

# Call the predict.py script with the correct arts
general_exp_kwargs  = {

    # 'llm_name': parse_kwargs.llm_name,
    # 'finetuned': parse_kwargs.finetuned,

    'predict_b2i': True,
    'predict_i2i': False,

    'prompt_style': prompt_style,
    'parse_style': parse_style,
    'effect_type': 'directly',

    # 'exp_group': parse_kwargs.exp_group,
    # 'exp_name': parse_kwargs.exp_name,
    
    'edge_value': 'binary_weight',

    # 'input_file':parse_kwargs.input_file,

    # "batch_size":parse_kwargs.batch_size,
    "ensemble_size": 1,
    "k_shot_b2i": 0,
    "k_shot_example_dset_name_b2i": None,
    "k_shot_example_dset_name_i2i": None,
    "k_shot_i2i": 0,

    "save_output":True,
    # "debugging":parse_kwargs.debugging,

    # "finetune_dir":parse_kwargs.finetune_dir,

    # 'use_system_prompt': parse_kwargs.use_system_prompt,
    # 'override_double_quant': parse_kwargs.override_double_quant,

    # 'unbias_categorisations': parse_kwargs.unbias_categorisations,
   
}



predict.main(
    **general_exp_kwargs,
    **vars(parse_kwargs)
)

# Open the saved file
dir_experiments = os.path.join('prompt_engineering','output','spot', parse_kwargs.exp_group )
existing_numbers = [int(x.split('_')[-1]) for x in os.listdir(dir_experiments) if x.startswith(f'exp_{parse_kwargs.exp_name}') ]
lowest_available_number = max(existing_numbers) if len(existing_numbers) > 0 else 0
experiment_number = lowest_available_number

save_dir = os.path.join(dir_experiments, f"exp_{parse_kwargs.exp_name}_{experiment_number:03d}" )

fn = os.path.join(save_dir, f'predictions_b2i.csv')

df = pd.read_csv(fn)

li_ensemble_predictions = df['predictions'].tolist()

li_scaled_entropy = []

for ensemble_predictions in li_ensemble_predictions:
    
    ensemble_predictions = eval(ensemble_predictions)

    ensemble_scaled_entropies = []

    for i in range(len(ensemble_predictions)):

        p_yes, p_no = ensemble_predictions[i]['Yes'], ensemble_predictions[i]['No']

        # normalize the predictions j.i.c
        p_yes = p_yes / (p_yes + p_no)
        p_no = p_no / (p_yes + p_no)

        entropy_val = entropy([p_yes, p_no], base=2)
        
        # Transform so more uncertainty is weaker weight
        entropy_val = 1 - entropy_val
        ensemble_scaled_entropies.append(entropy_val)
    
    avg_scaled_entropy = sum(ensemble_scaled_entropies) / len(ensemble_scaled_entropies)

    li_scaled_entropy.append(avg_scaled_entropy)

df['edge_existence_scaled_entropy'] = li_scaled_entropy
df.to_csv(fn, index=False)