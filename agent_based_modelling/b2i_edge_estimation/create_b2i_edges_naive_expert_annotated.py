import os
import pandas as pd

# Reading the file connecting broad budget items to indicators
df_indicators = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_sample_raw.csv'), usecols=['seriesCode','seriesName', 'category1','category2','category3', 'Age', 'Sex'])
df_indicators = df_indicators.rename({'seriesCode':'indicator_code', 'seriesName':'indicator', 'category1':'broad_budget_item1', 'category2':'broad_budget_item2', 'category3':'broad_budget_item3'}, axis=1)

# Reading the file connecting broad budget items to fine grained budget items
df_finegrained_budget_items = pd.read_csv(os.path.join('data','ppi','data_expenditure_raw.csv'), usecols=['seriesCode','seriesName', 'category']).rename({'seriesName':'budget_item_finegrained',
'category':'broad_budget_item', 'seriesCode':'bi_fg_code'}, axis=1)

# Collapse the df_indicators broad_budget_items_n columns so remaining columns are ['seriesCode', 'indicator', 'broad_budget_item'].
# With one row for each indicator and broad_budget_itemcombination.
df_indicators = pd.melt(df_indicators, id_vars=['indicator_code', 'indicator'], value_vars=['broad_budget_item1', 'broad_budget_item2', 'broad_budget_item3'], value_name='broad_budget_item').drop('variable', axis=1)

# Merge the df_indicators and df_finegrained_budget_items on the broad_budget_itemcolumn
df_naive_b2i_finegrained = df_indicators.merge(df_finegrained_budget_items, on='broad_budget_item')

# Save to naive_b2i_finegrained.csv
df_naive_b2i_finegrained.to_csv(os.path.join('data','ppi', 'b2i_networks','naive_b2i_finegrained.csv'), index=False)