# This script is designed to impute missing indicator values for future years based on the 
# Public Policy Intervention (PPI) model, utilizing both agent-based modeling and network analysis 
# techniques. It performs the following key functions:
# 1. Loads model parameters and hyperparameters from previously calibrated experiments, 
#    allowing for the customization of the imputation process based on experimental setups.
# 2. Processes input data including current indicator levels, forecasted resource allocations, 
#    and legal frameworks to prepare for simulation.
# 3. Utilizes the PPI model to forecast indicator values over a specified number of years, 
#    starting from a given year. The model supports both serial and parallel execution modes, 
#    facilitating Monte Carlo simulations for uncertainty analysis.
# 4. Constructs and utilizes both budget-to-indicator (B2I) and indicator-to-indicator (I2I) 
#    networks based on specified methods and thresholds, enhancing the model's ability to 
#    simulate complex policy impacts and spillover effects.
# 5. Aligns forecasted budgets with the B2I network to ensure consistency in the simulation inputs.
# 6. Saves the imputed indicators, resource allocations, and spillover effects for further analysis 
#    and comparison with actual indicator values, aiding in the evaluation of the model's performance.
# 7. Offers flexibility in specifying the experiment group and number for both loading calibration 
#    parameters and saving imputation results, supporting structured experimentation and analysis.

# Usage of the script is facilitated through command-line arguments, making it adaptable to different 
# scenarios and datasets. The script stands as a comprehensive tool for researchers and policymakers 
# to analyze the potential outcomes of public policies through advanced simulation techniques.

import argparse
import pandas as pd
import os
import numpy as np
import yaml
from agent_based_modelling.ppi import run_ppi, run_ppi_parallel, align_Bs_with_B_dict
import glob
import pickle
import logging
from agent_based_modelling.calibration import get_b2i_network, get_i2i_network


def main( impute_start_year:int=2018, impute_years:int=1,
            exp_group:str|None=None, exp_num:int=0,
            mc_simulations:int=1, parallel_processes:int|None=None,
            i2i_threshold:float|None=None,
            save_exp_group:str|None=None,
            save_exp_num:int|None=None):
    """
    Impute the missing indicator values for the next n time steps using the PPI model.  
    
    Parameters: 
    - impute_start_year: The year to start imputing from
    - impute_years: Number of years to impute. This assumes you have the final indicator level after the periods to be imputed
    - exp_group: The name of the experiment group. Used to load the calibrated parameters. Also used to save the imputations
    - exp_num: The experiment number. Used to load the calibrated parameters. Also used to save the imputations 
    - mc_simulations: Number of Monte Carlo simulations to run
    - parallel_processes: Number of parallel processes to run
    - i2i_threshold: The min value an i2i edge weight must have to be included in the i2i network. If None, all i2i edges are included.
    - save_exp_group: The name of the experiment group to save the imputations to. Overrides the exp_group argument.
    - save_exp_num: The experiment number to save the imputations to. Overrides the exp_num argument.
    """

    # Load parameters from trained ppi model
    # Load calibration_kwargs e.g. the params for the PPI model
    model_params = load_model_kwargs( exp_num, exp_group )
    model_hparams = load_model_hparams( exp_num, exp_group )

    current_I, fBs, frl, fG, time_refinement_factor, impute_periods = load_currI_fBs_frl_fG( impute_start_year=impute_start_year, impute_years=impute_years,
                                                                                exp_group=exp_group, exp_num=exp_num  )

    i2i_network = get_i2i_network( model_hparams['i2i_method'], current_I.shape[0], model_hparams['model_size'], i2i_threshold=i2i_threshold )
    
    b2i_network = get_b2i_network( model_hparams['b2i_method'], model_hparams['model_size'] )

    Bs, b2i_network = align_Bs_with_B_dict(fBs, b2i_network)

    impute_output = impute_indicators( impute_years, time_refinement_factor,
                                                current_I, fBs, frl, fG, i2i_network, 
                                                b2i_network, model_params = model_params,
                                                parallel_processes=parallel_processes,
                                                mc_simulations=mc_simulations,
                                                adjusted_impute_periods=impute_periods )
    
    indicator_values, indicator_names = load_true_indicators( impute_years=impute_years, impute_start_year=impute_start_year )

    #  Save the imputed and true indicators to file
    outp = {
        'imputed_indicators': impute_output['imputed_indicators'],
        'imputed_allocations': impute_output['imputed_allocations'],
        'imputed_spillovers': impute_output['imputed_spillovers'],
        'target_indicators': indicator_values,
        'impute_start_year': impute_start_year,
        'impute_years': impute_years,
        'indicator_names': indicator_names,
        'exp_num': exp_num,
        'exp_group': exp_group,
        'save_exp_num': save_exp_num if save_exp_num is not None else exp_num,
        'save_exp_group': save_exp_group if save_exp_group is not None else exp_group,
        'mc_simulations': mc_simulations,
        'i2i_threshold': i2i_threshold,
        'model_hparams': model_hparams,
    }


    save_dir = os.path.join('.','agent_based_modelling','output', 'imputations', f'{save_exp_group if save_exp_group else exp_group}' )
    os.makedirs(save_dir, exist_ok=True)
        
    fn = f'exp_{str(save_exp_num if save_exp_num else exp_num).zfill(3)}.pkl'
    with open(os.path.join(save_dir, fn), 'wb') as f:
        pickle.dump(outp, f)

def load_model_kwargs( exp_num:int, exp_group=None ) -> pd.DataFrame:
    
    if exp_group is None:
        f_pattern = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}','params_v**.csv') 
    else:
        f_pattern = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', exp_group, f'exp_{str(exp_num).zfill(3)}','params_v**.csv')

    # get list of versions of parameters associated with the experiment number
    param_files = glob.glob( f_pattern )

    # get the latest version - this should the file with the highest goodness of fit
    fp = sorted(param_files)[-1]

    df_parameters = pd.read_csv(fp)

    return df_parameters

def load_model_hparams( exp_num, exp_group=None ) -> pd.DataFrame:
    
    if exp_group is None:
        f_pattern = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}','hyperparams.yaml')
    else:
        f_pattern = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', exp_group, f'exp_{str(exp_num).zfill(3)}','hyperparams.yaml')

    # get list of versions of parameters associated with the experiment number
    param_files = glob.glob( f_pattern )

    # get the latest version - this should the file with the highest goodness of fit
    fp = sorted(param_files)[-1]

    df_parameters = yaml.safe_load( open(fp ,"r") )

    return df_parameters

def load_currI_fBs_frl_fG(impute_start_year=2018, impute_years=1, exp_group=None, exp_num=0):
    """
    Load the current indicator levels, forecasted resource allocation, and forecasted rule of law for the next n time steps.
    """

    # exp_dir = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}' )
    # # get list of versions of parameters associated with the experiment number
    # # get the latest version - this should the file with the highest goodness of fit
    # f_pattern_params = os.path.join(exp_dir, 'params_v**.csv')
    # param_files = glob.glob( f_pattern_params )
    # fp = sorted(param_files)[-1]
    # calibration_params = pd.read_csv(fp)

    if exp_group is None:
        calibration_hparams = os.path.join( '.', 'agent_based_modelling', 'output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}', 'hyperparams.yaml')
    else:
        calibration_hparams = os.path.join( '.', 'agent_based_modelling', 'output', 'calibrated_parameters', exp_group, f'exp_{str(exp_num).zfill(3)}', 'hyperparams.yaml')

    calibration_hparams = yaml.safe_load( open( calibration_hparams, 'r' ) )

    calibration_start_year = calibration_hparams['calibration_start_year']
    calibration_end_year = calibration_hparams['calibration_end_year']
    
    # The start and final year are used as inputs to the PPI model
    impute_final_year = impute_start_year + impute_years -1

    # Load the data
    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
    df_exp = pd.read_csv('./data/ppi/pipeline_expenditure_finegrained.csv')
    
    years = [col for col in df_indic.columns if str(col).isnumeric() ] #if col>=calibration_start_year and col<=impute_final_year ]
    years_int = [int(col) for col in years]
    tft = time_refinement_factor = df_exp['time_refinement_factor'].values[0]

    # Checking that the forecast period is within the bounds of the training data
    
    assert ( str(int(impute_start_year-1)) in years) and (str(impute_final_year) in years), \
        f'Impute period is not within the bounds of the available data'

    # current_I
    # we take imputation levels from the end of previous year
    current_I = df_indic[ str(impute_start_year-1) ].values

    # fBs - control budget allocation
    # Our imputation periods starts from one interpolation into the imputation start year
    impute_start_period_idx = ( years_int.index(impute_start_year)-years_int.index(calibration_start_year) )*tft
    impute_final_period_idx = ( years_int.index(impute_final_year)-years_int.index(calibration_start_year) )*tft + (tft-1)
    fBs_cols = [ str(idx) for idx in  range(impute_start_period_idx, impute_final_period_idx+1) ]
    fBs = df_exp[ fBs_cols ].values

    # fR
    frl = df_indic.rl.values # quality of the rule of law

    # fG
    fG = df_indic[str(impute_final_year)].values

    # 
    impute_periods = len(fBs_cols)

    return current_I, fBs, frl, fG, time_refinement_factor, impute_periods
    
def impute_indicators(impute_years, time_refinement_factor, current_I, fBs, frl, fG, 
    i2i_network, b2i_network, model_params, parallel_processes=None, mc_simulations=1,
    adjusted_impute_periods=None ):
    """
    Forecast the indicator levels for the next n time steps using the PPI model.
    
    Parameters:
    - filepath: The path to the saved parameter file
    - current_I: The current indicator levels
    - P: Resource allocation for each indicator for the next n time steps
    - fBs: Forecasted Budget Allocation for the next n time steps
    - fR: Projected rule of law for the next n time steps
    
    Returns:
    - Forecasted indicator levels for the next n time steps
    """
    if adjusted_impute_periods is not None:
        impute_periods = adjusted_impute_periods
    else:
        impute_periods = impute_years*time_refinement_factor

    assert fBs.shape[1] == impute_periods, f'fBs must be an array of shape (budget_item_count, {impute_periods} )'
    assert len(frl) == len(current_I) , f'fR must have an element for each indicator. fR has {len(frl)} elements, while current_I has {len(current_I)} elements'
    
    # Extract the controlled parameters
    I0 = current_I # Initial indicator levels
    T = impute_periods # Forecast period
    G = fG # Target indicator level
    rl = frl 
    Bs = fBs # Forecasted Budget Allocation

    so_network = i2i_network
    B_dict = b2i_network

    # Extract the necessary parameters
    alphas = model_params['alpha'].values
    alphas_prime = model_params['alpha_prime'].values
    betas = model_params['beta'].values
    
    Imax = model_params.get('Imax', None)
    Imin = model_params.get('Imin', None)

    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 

    R = df_indic.R.values
    qm = df_indic.qm.values
    bs = df_indic.bs.values if 'bs' in df_indic.columns else None
    
    # Step 2 and 4 are combined: Run the model for frl steps, 
    # using the specified P and fR as inputs
    if mc_simulations==1 and (parallel_processes is None or parallel_processes == 1):
        imputed_indicators, _, _, imputed_allocations, imputed_spillovers, _ = run_ppi(I0=I0, alphas=alphas, alphas_prime=alphas_prime,
                                 betas=betas, so_network=so_network, R=R, 
                                 bs=bs, qm=qm, rl=rl,
                                 Imax=Imax, Imin=Imin, 
                                 Bs=Bs, B_dict=B_dict, G=G,
                                 T=impute_periods)
    else:
        li_imputed_indicators, _, _, li_imputed_allocations, li_imputed_spillovers, _ = run_ppi_parallel(I0=I0, alphas=alphas, alphas_prime=alphas_prime,
                                 betas=betas, so_network=so_network, R=R, 
                                 bs=bs, qm=qm, rl=rl,
                                 Imax=Imax, Imin=Imin, 
                                 Bs=Bs, B_dict=B_dict, G=G,
                                 T=impute_periods,
                                 parallel_processes=max(parallel_processes,1),
                                 sample_size=mc_simulations)
        
        imputed_indicators = np.stack(li_imputed_indicators, axis=0).swapaxes(1,2) # (mc_simulations, impute_periods, indicator_count)
        imputed_allocations = np.stack(li_imputed_allocations, axis=0).swapaxes(1,2) # (mc_simulations, impute_periods, indicator_count)
        imputed_spillovers = np.stack(li_imputed_spillovers, axis=0).swapaxes(1,2) # (mc_simulations, impute_periods, indicator_count, indicator_count)

    

    # Handling the time_refinement_factor
    # The indicator time series is interpolated to a finer time scale of factor time_refinement_factor
    if time_refinement_factor > 1:

        imputed_indicators = imputed_indicators[:, ::-time_refinement_factor ][:, ::-1] 
        imputed_allocations = imputed_allocations[:, ::-time_refinement_factor ][:, ::-1] 
        imputed_spillovers = imputed_spillovers[:, ::-time_refinement_factor ][:, ::-1] 
    
    return {'imputed_indicators': imputed_indicators, 'imputed_allocations': imputed_allocations, 'imputed_spillovers': imputed_spillovers}

def load_true_indicators( impute_start_year, impute_years):
    """
    Load the true indicator levels for the next forecast_periods time steps.
    """

    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
    indicator_names = df_indic.indicator_name.values
    indicator_values = df_indic[ [str(year) for year in range(impute_start_year, impute_start_year+impute_years) ] ].values.T
    # hyper_params = yaml.safe_load( open( os.path.join(save_dir, 'hyperparams.yaml'), 'r' ) )
    
    return indicator_values, indicator_names

def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Impute missing values in a time series.')
    parser.add_argument('--impute_start_year', type=int, default=2018, help='The year to start imputing from')
    parser.add_argument('--impute_years', type=int, default=1, help='Number of years to impute. This assumes you have the final indicator level after the periods to be imputed')
    parser.add_argument('--exp_group', type=str, default=None, help='The name of the experiment group')
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--save_exp_group', type=str, default=None, help='The name of the experiment group to save the imputations to. Overrides the exp_group argument.')
    parser.add_argument('--save_exp_num', type=int, default=None, help='The experiment number to save the imputations to. Overrides the exp_num argument.')
    parser.add_argument('--mc_simulations', type=int, default=1, help='Number of Monte Carlo simulations to run')
    parser.add_argument('--parallel_processes', type=int, default=None, help='Number of parallel processes to run')
    parser.add_argument('--i2i_threshold', type=float, default=None, help='The min value an i2i edge weight must have \
        to be included in the i2i network. If None, all i2i edges are included')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()   
    
    main(**vars(args))
    