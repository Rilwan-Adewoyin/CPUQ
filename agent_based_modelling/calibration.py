# This script implements the calibration and analysis pipeline for a policy performance impact (PPI) model. It integrates various components such as agent-based modeling,
# data preprocessing, and logging to assess the impact of policy interventions on different indicators over a specified time frame. The script supports different methods for
# budget-to-indicator (b2i) and indicator-to-indicator (i2i) relationship estimation, leveraging machine learning models for predicting these relationships. It enables the user
# to specify a range of parameters, including the start and end years for calibration, the model size, thresholds for calibration, and options for running the model with different
# levels of verbosity and debugging. The script also provides functionalities for parallel processing to accelerate the calibration process, handling of low precision counts to
# refine calibration, and the capability to conduct multiple Monte Carlo simulations to ensure robustness in the calibration results. Additionally, it includes utilities for logging
# and output management, ensuring that the calibration results, parameters, and runtime statistics are systematically recorded for analysis. The argparse library is used for
# command-line argument parsing, allowing for flexible and user-defined model configuration.

import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse 
from agent_based_modelling import ppi
from utils import ALL_MODELS
import yaml
warnings.simplefilter("ignore")
import glob
from collections import defaultdict, Counter

from agent_based_modelling.i2i_edge_estimation.create_i2i_candidates import concatenate_name_age
from copy import deepcopy
from prompt_engineering.my_logger import setup_logging_calibration
logging = None
from builtins import FileNotFoundError
from typing import Dict
import math
#TODO: ensure the b2i method is handled e.g. figure out what edits to do in order to get the b2i matrix

def main(   start_year, end_year,
            parallel_processes,
            b2i_method, i2i_method, model_size,
            thresholds:list[float], low_precision_counts, increment,
            debugging, time_experiments=False,
            mc_simulations=3,
            exp_samples=1,
            exp_group = None,
            verbose=False ):

    global logging
    logging =  setup_logging_calibration(debugging=debugging, exp_group=exp_group)
    
    # Log parameters for this experiment
    logging.info("Parameters for this experiment:")
    logging.info(f"\tstart_year: {start_year}")
    logging.info(f"\tend_year: {end_year}")
    logging.info(f"\tparallel_processes: {parallel_processes}")
    logging.info(f"\tthresholds: {thresholds}")
    logging.info(f"\tlow_precision_counts: {low_precision_counts}")
    logging.info(f"\tmc_simulations: {mc_simulations}")
    logging.info(f"\tincrement: {increment}")
    logging.info(f"\tb2i_method: {b2i_method}")
    logging.info(f"\ti2i_method: {i2i_method}")
    logging.info(f"\tmodel_size: {model_size}")
    logging.info(f"\tverbose: {verbose}")
    logging.info(f"\ttime_experiments: {time_experiments}")

    # Get calibration kwargs
    calibration_kwargs = get_calibration_kwargs(b2i_method,
                                                i2i_method,
                                                model_size,
                                                calibration_start_year=start_year,
                                                calibration_end_year=end_year)

    exp_dir = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters' )
    if exp_group is not None:
        exp_dir = os.path.join(exp_dir, exp_group)
    os.makedirs(exp_dir, exist_ok=True)

    # Create seperate experiment directory for each threshold
    for threshold in thresholds:
        
        logging.info(f'Calibrating with threshold {threshold}')
        
        # TODO: if time-experiments is set collect information on #1) experiment time to complete 2) number of steps takn
        li_exp_output:list[dict] = []
        
        for run in range(exp_samples):
            logging.info(f'Calibration run {run+1} of {exp_samples}')

            dict_output = calibrate( low_precision_counts=low_precision_counts,
                    threshold=threshold,
                    parallel_processes=parallel_processes,
                    verbose=verbose,
                    time_experiments=time_experiments,
                    increment=increment,
                    mc_simulations=mc_simulations,
                    **calibration_kwargs )

            li_exp_output.append(dict_output)
            logging.info(f'Calibration run {run+1} of {exp_samples} complete')
        
        logging.info(f'Calibration with threshold {threshold} complete')

        # Create experiment number which accounts for any missing numbers in the sequence and selects the lowest possible number available
        existing_exp_numbers = [int(exp_number[4:]) for exp_number in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, exp_number))]
        existing_exp_numbers.sort()
        if len(existing_exp_numbers) == 0:
            exp_number = 0
        else:
            exp_number = next( (num for num in range(existing_exp_numbers[-1]+1) if num not in existing_exp_numbers ), existing_exp_numbers[-1]+1)
        exp_number_str = f'exp_{str(exp_number).zfill(3)}'
        os.makedirs( os.path.join(exp_dir, exp_number_str), exist_ok=True)
        
        # Saving parameters for each sample run
        for idx, dict_output in enumerate(li_exp_output):
            
            _ = deepcopy(dict_output['parameters'])
            _['threshold'] = threshold
            df_parameters = pd.DataFrame(_)
            df_parameters.to_csv(os.path.join(exp_dir, exp_number_str, f'params_v{ str(idx).zfill(2) }.csv'), index=False)
        
        # Saving statistics on run times
        if time_experiments:
            time_elapsed:list[int] = [dict_output['time_elapsed'] for dict_output in li_exp_output]
            iterations:list[int] = [dict_output['iterations'] for dict_output in li_exp_output]
            
            df_time = pd.DataFrame({'time_elapsed':time_elapsed, 'train_iters':iterations, 'sample_number':list(range(exp_samples) ) })
            df_time.to_csv(os.path.join(exp_dir, exp_number_str, f'calibration_time.csv'), index=False)

        # Save hyperparameters as yaml
        experiment_hyperparams = {
            'calibration_start_year':start_year,
            'calibration_end_year':end_year,
            'parallel_processes':parallel_processes,
            'threshold': threshold,
            'low_precision_counts':low_precision_counts,
            'mc_simulations':mc_simulations,
            'increment':increment,
            'b2i_method':b2i_method,
            'i2i_method':i2i_method,
            'model_size':model_size,
            'exp_number':exp_number,
            'time_experiments':time_experiments
        }

        with open(os.path.join(exp_dir, exp_number_str, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(experiment_hyperparams, f)

    return True

def get_calibration_kwargs(
                        b2i_method,
                        i2i_method, 
                        model_size,
                        calibration_start_year=2014,
                        calibration_end_year=2017,
                            ):

        
    if calibration_start_year == 2014 and calibration_end_year == 2017:
        df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
        df_exp = pd.read_csv('./data/ppi/pipeline_expenditure_finegrained.csv') 
    else:
        raise NotImplementedError('Calibration files only created for 2014-2017')

    colYears_train = [col for col in df_indic.columns if str(col).isnumeric() if int(col)>=calibration_start_year and int(col)<=calibration_end_year] 
    tft = df_exp.time_refinement_factor.values[0]
    

    # Only Calibrate On The Interpolated Periods between 2014 to 2017
    # This is from 2014 Interpolation period 0 to 2017 Interpolation period 4,
    # Where each year includes N interpolation periods
    # We don't start from 2014 Interpolation period 0, since our indicator_start is the value for 2013 
    expCols = [col for col in df_exp.columns if str(col).isnumeric()]
    s_idx = colYears_train.index(str(calibration_start_year))*tft
    e_idx = (colYears_train.index(str(calibration_end_year))+1)*tft
    assert e_idx <= int(expCols[-1]) +1 , f'End period index {e_idx} is greater than the number of periods {len(expCols)}'
    expCols_train = expCols[ s_idx : e_idx ]

    T = len(expCols_train) #Timesteps

    indic_count = len(df_indic) # number of indicators
    indic_start = df_indic[ str(int(colYears_train[0])-1)].values
    indic_final = df_indic[colYears_train[-1]].values
    
    
    success_rates = df_indic.successRates.values # success rates
    R = np.ones(indic_count) # instrumental indicators
    qm = df_indic.qm.values # quality of monitoring
    rl = df_indic.rl.values # quality of the rule of law

    Bs = df_exp[expCols].values # disbursement schedule (assumes that the expenditure programmes are properly sorted)
    
    b2i_network = get_b2i_network( b2i_method, model_size )

    Bs, b2i_network = ppi.align_Bs_with_B_dict(Bs, b2i_network)

    # TODO: Handling cases where some budget items have not been linked to any indicators


    # Load in the i2i relation table
    i2i_network = get_i2i_network( i2i_method=i2i_method, model_size=model_size, indic_count=indic_count )


    return {
        'indic_start': indic_start,
        'indic_final': indic_final,
        'success_rates': success_rates,
        'R': R,
        'qm': qm,
        'rl': rl,
        'Bs': Bs,
        'B_dict': b2i_network,
        'T': T,
        # 'indic_count': indic_count,
        # 'indis_index': indis_index,
        'i2i_network':i2i_network
    }

def get_b2i_network(b2i_method,  model_size) -> dict[int, list[int]]:
    # Create a dictionary which aligns indicators with budget items
    # Both indicators and budget items are referred to by their index in data_expenditure_trend_finegrained and data_expenditure_raw respectively
    # B = { 'idic_idx0':[bi_idx0, bi_idx1, ... ] }
    if b2i_method in ['verbalize', 'CPUQ_binomial']:
        assert model_size is not None, 'model_size must be specified when using verbalize or CPUQ_binomial'

    if b2i_method == 'ea':
        # Load in the b2i relation table
        df_rela = pd.read_csv(os.path.join('data','ppi','pipeline_relation_table_finegrained.csv'))

        B_dict = {} # PPI needs the relational table in the form of a Python dictionary
        for index, row in df_rela.iterrows():
            B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]

    elif b2i_method in ['verbalize', 'CPUQ_binomial']:
        
        # Load the varbalize
        if b2i_method == 'verbalize':
            dir_ = os.path.join('prompt_engineering', 'output', 'spot', f'ppi_b2i_{model_size}_verbalize')
        elif b2i_method == 'CPUQ_binomial':
            dir_ = os.path.join('prompt_engineering', 'output', 'spot', f'ppi_b2i_{model_size}_cpuq')

        _ =  glob.glob(os.path.join(dir_, '**' ,'**predictions_b2i.csv')) # (budget_item,indicator, related, pred_aggregated, prompts, predictions, discourse)
        if len(_) == 0:
            raise FileNotFoundError(f'No b2i predictions found at {dir_}')
        b2i_preds = pd.read_csv(_[0])
        b2i_preds['pred_aggregated'] = b2i_preds['pred_aggregated'].apply(lambda x: eval(x))
        # filter b2i_preds on rows where pred_aggregated dict has key 'Yes' with value more than 0.5
        b2i_preds = b2i_preds[ b2i_preds.pred_aggregated.apply(lambda x: x.get('Yes', 0.0) >= 0.5) ]

        # load info on the index ordering of indicators

        # Create Map of indicator index to indicator name
        pipeline_indicators = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_normalized_finegrained.csv'), usecols=['indicator_name'])
        dict_indic_idx = {v: k for k, v in pipeline_indicators['indicator_name'].to_dict().items()}
        
        # Info on non unique indicators (by name)
        dict_idx_indic = pipeline_indicators['indicator_name'].to_dict()
        indicator_list = list(dict_idx_indic.values())
        non_unique_indicators = [item for item, count in Counter(indicator_list).items() if count > 1]

        # Create Map of budget item index to budget item name
        pipeline_budget_items = pd.read_csv(os.path.join('data','ppi','pipeline_expenditure_finegrained.csv'), usecols=['seriesName'])
        dict_bi_idx = {v: k for k, v in pipeline_budget_items['seriesName'].to_dict().items()}

        # Info on non unique budget items (by name)
        dict_idx_bi  = pipeline_budget_items['seriesName'].to_dict() 
        non_unique_bi = [item for item, count in Counter(dict_idx_bi.values()).items() if count > 1]
        

        B_dict = {idx:[] for idx in range(len(indicator_list))} # PPI needs the relational table in the form of a Python dictionary

        for _, row in b2i_preds.iterrows():
            budget_item = row.budget_item
            indicator = row.indicator
            
            # Removing predictions which are not needed in the pipeline
            if budget_item not in dict_bi_idx or indicator not in indicator_list:
                continue
            
            # Handling cases where budget items have the same name
                # If a formatted budget item name is shared by two different budget items then
                # We ensure both budget item indexes get assigned the indicator index   
            if budget_item in non_unique_bi:
                li_bi_idx = [idx for idx, bi in dict_idx_bi.items() if bi==budget_item]
            else:
                li_bi_idx = [ dict_bi_idx[ budget_item ] ] # get the index of the indicator

            # Handling cases where indicators have the same name
                # If a formatted indicator name is shared by multiple different indicators then 
                # We ensure both indicator indexes get assigned the budget item index            
            if indicator in non_unique_indicators:
                li_ind_idx = [ i for i,indic in dict_idx_indic.items() if indic==indicator]
            else:
                li_ind_idx = [ dict_indic_idx[ indicator ] ] # get the index of the indicator
                
            for ind_idx in li_ind_idx:
                for bi_idx in li_bi_idx:
                    B_dict[ind_idx].append(bi_idx)

        # For any indicators that have not been assigned an associated expenditure programme, use the default value from 'ea' method
        df_rela = pd.read_csv(os.path.join('data','ppi','pipeline_relation_table_finegrained.csv'))
        for index, row in df_rela.iterrows():
            if len(B_dict[int(row.indicator_index)]) == 0:
                B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]
    return B_dict

def get_i2i_network(i2i_method, indic_count, model_size=None, i2i_threshold=None):
    # Creates an array representing indicator to indicator relationships

    if i2i_method in ['verbalize', 'CPUQ_multinomial', 'CPUQ_multinomial_adj','verbalize','entropy']:
        assert model_size is not None, 'model_size must be specified when using verbalize or CPUQ_multinomial'
    if i2i_threshold is not None:
        assert i2i_method in ['CPUQ_multinomial', 'CPUQ_multinomial_adj', 'verbalize', 'ccdr' ], 'Can only filter out edges by i2i_threshold when using CPUQ_multinomial, CPUQ_multinomial_adj, verbalize or entropy'
    
    i2i_network = None

    if i2i_method == 'zero':
        i2i_network = np.zeros((indic_count, indic_count))

    elif i2i_method == 'ccdr':
        _path = os.path.join('data','ppi','i2i_networks','ccdr.csv')
        if os.path.exists(_path):
            df_net = pd.read_csv(_path)
        else:
            raise FileNotFoundError(f'i2i network not created - no file at {_path}')

        i2i_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        for index, row in df_net.iterrows():
            
            if row.Weight < i2i_threshold:
                continue

            i = int(row.From)
            j = int(row.To)
            w = row.Weight
            i2i_network[i,j] = w

    elif i2i_method in [ 'CPUQ_multinomial', 'CPUQ_multinomial_adj' ,'verbalize', 'entropy']:

        # Reading in the i2i network
        _1 = {
            '7bn':os.path.join('exp_i2i_7bn_distr','exp_sbeluga7b_non_uc'),
            '13bn':os.path.join('exp_i2i_13b_distr','exp_sbeluga13b_non_uc'),
            '30bn':os.path.join('exp_i2i_30b_distr','exp_upllama30b_non_uc'),
        }

        _2 = {'CPUQ_multinomial':'mn', 'verbalize':'vb', 'entropy':'et', 'CPUQ_multinomial_adj':'mn_adjusted'}
        
        _path = os.path.join('prompt_engineering','output', 'spot', _1[model_size], 
            f'i2i_{_2[i2i_method]}_weights.csv')
        
        if os.path.exists(_path):
            df_net = pd.read_csv(_path) # columns = ['indicator1', 'indicator2', 'weight']
            # filtering out rows where no weight predict or weight was None
            df_net = df_net[ ~df_net.weight.isnull() ]
            if not isinstance(df_net.weight[0], dict):
                df_net.weight =  df_net.weight.apply(lambda x: eval(x))
        else:
            raise FileNotFoundError(f'i2i Network not created - no file at {_path}')
        

        # convert to a dictionary where the key is the indicator_name and the value is the indicator's index
        indicator_ref = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_normalized_finegrained.csv'), usecols=['indicator_name'])
        dict_indic_idx = {v: k for k, v in indicator_ref['indicator_name'].to_dict().items()}

        i2i_network = np.zeros((indic_count, indic_count), dtype=np.float32) # adjacency matrix
        
        for index, row in df_net.iterrows():
            
            weight = row.weight.get('scaled_mean', row.weight.get('mean', 0.0))
            
            if weight == 0.0:
                continue
            
            # thresholding
            if i2i_threshold is not None and 'distribution' in df_net.columns:
                # Filtering out edges with entropy above threshold
                
                if i2i_method in ['CPUQ_multinomial', 'CPUQ_multinomial_adj']:
                    distribution : dict = row.distribution[0] if isinstance(row.distribution, list) else eval(row.distribution)[0]
                    entropy = interpretable_entropy( distribution  )
                    if entropy < i2i_threshold:
                        continue
                elif i2i_method in ['verbalize']:
                    if weight < i2i_threshold:
                        continue

            i = dict_indic_idx[row.indicator1] 
            j = dict_indic_idx[row.indicator2] 
            
            i2i_network[i,j] = weight
        
    return i2i_network

def interpretable_entropy(distr:Dict[int,float] ) -> float:
    base = len(distr)
    entropy = 1 + sum( [ p * math.log(p) * (1/math.log(base)) if p!=0 else 0.0 for p in distr.values() ] )
    return entropy  

def calibrate(indic_start, indic_final, success_rates, R, qm, rl, Bs, B_dict, T, i2i_network,
              parallel_processes=6, threshold=0.8,
              low_precision_counts=75,
              increment=100,
              verbose=True, 
              time_experiments=False,
              mc_simulations=10):

    dict_output = ppi.calibrate(indic_start, indic_final, success_rates, so_network=i2i_network, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
                T=T, threshold=threshold, parallel_processes=parallel_processes, verbose=verbose,
                low_precision_counts=low_precision_counts, time_experiments=time_experiments,
                increment=increment,
                mc_simulations=mc_simulations, 
                logging=logging)

    return dict_output

# Create an argparse function to parse args
def get_args():
    parser = argparse.ArgumentParser(description='Run the PPI model')
    parser.add_argument('--start_year', type=int, default=2014, help='Start year')
    parser.add_argument('--end_year', type=int, default=2017, help='End year')
    parser.add_argument('--parallel_processes', type=int, default=40, help='Number of parallel processes')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.8], help='Threshold for the calibration')
    parser.add_argument('--low_precision_counts', type=int, default=75, help='Number of low-quality iterations to accelerate the calibration')
    parser.add_argument('--increment', type=int, default=100, help='Number of iterations between each calibration check')

    parser.add_argument('--mc_simulations', type=int, default=5, help='Number of Monte Carlo simulations to run for each iteration')

    parser.add_argument('--b2i_method', type=str, default='ccdr', choices=[ 'ea', 'verbalize', 'CPUQ_binomial' ], help='Name of the spillover predictor model')
    parser.add_argument('--i2i_method', type=str, default='ccdr', choices=['ccdr', 'verbalize' ,'CPUQ_multinomial', 'CPUQ_multinomial_adj' ,'zero', 'entropy'], help='Name of the indicator to indicator edge predictor method')
    parser.add_argument('--model_size', type=str, default=None, choices=['7bn','13bn','30bn' ], help='Name of the indicator to indicator edge predictor method')
    
    parser.add_argument('--verbose', action='store_true', default=False, help='Print progress to console')
    parser.add_argument('--time_experiments', action='store_true', default=False, help='Record Calibration Time for Experiments')
    parser.add_argument('--exp_samples',type=int, default=1, help='Number of samples to take for time experiments')
    parser.add_argument('--exp_group', type=str, default=None, help='Name of the experiment group')
    parser.add_argument('--debugging', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # pass args in as kwargs
    main(**vars(args))
    