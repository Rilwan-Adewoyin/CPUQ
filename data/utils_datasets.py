import pandas as pd
import os
from datamodels.healthindicators import GovernmentHealthIndicatorSet
from datamodels.governmentbudget import GovernmentBudget

dict_paths = {
    'england':{'budgetitems': os.path.join('england','processed_datasets','budget_items.csv') }
}

def load_budget_dataset( dset_name ):

    fp_budget_items = os.path.join('datasets',dset_name,'processed_datasets','budget_items.csv')
    df_budget_items = pd.read_csv(fp_budget_items, header=0)

    return df_budget_items
    

def load_health_indicators_dset(dset_name):

    fp_health_indicators = os.path.join('datasets',dset_name,'processed_datasets','health_indicators.csv')
    df_health_indicators = pd.read_csv(fp_health_indicators, header=0)
    return df_health_indicators