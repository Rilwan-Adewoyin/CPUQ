import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from prompt_engineering.utils_prompteng import create_negative_examples_b2i
from sklearn.model_selection import train_test_split

"""
    This script creates the train and test splits for the SPOT dataset.
"""

remove_public_health = True
random_state_seed = 10


dset = pd.read_csv('./data/spot/spot_indicator_mapping_table.csv')

# Remove all rows where 'type' is not 'Outcome'
dset = dset[dset['type'] == 'Outcome']

# Remove all rows where 'category' 

# Creating target field
dset['related'] = 'Yes'

# Rename columns to match the format of the other datasets
dset = dset.rename( columns={'category': 'budget_item', 'name':'indicator' } )

# Create negative examples
random_state = np.random.RandomState(random_state_seed)

# remove all rows from dset that have value 'Public Health' for budget_item
if remove_public_health:
    dset = dset[ dset['budget_item'] != 'Public Health' ]

# To vauge a budget item and there are only 4 examples of it, we remove it
# dset = dset[ dset['budget_item'] != 'Central' ]

# Replace budget_item with 'Central' with 'Central Services'
dset['budget_item'] = dset['budget_item'].replace('Central', 'Central Services')

# create negative examples
dset = create_negative_examples_b2i(dset, random_state=random_state )

# Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
dset = dset.groupby(['budget_item','related']).filter(lambda x: len(x) > 1)

# perform stratified split of a dataframe into train and test subsets
train_dset, test_dset = train_test_split(dset, test_size=0.8, random_state=random_state, stratify=dset[['budget_item','related']])

# save train and test splits
train_dset.to_csv('./data/spot/spot_b2i_broad_train.csv', index=False)
test_dset.to_csv('./data/spot/spot_b2i_broad_test.csv', index=False)