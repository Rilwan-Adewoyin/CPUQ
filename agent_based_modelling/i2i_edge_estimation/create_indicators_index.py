import pandas as pd
from create_i2i_candidates import concatenate_name_age

# Load the indicators file
df_indic = pd.read_csv('data/ppi/pipeline_indicators_normalized_2013_2016.csv')

# create an index column which is 0-based
df_indic['idx'] = df_indic.index
df_indic['indicator_name_fmtd'] = concatenate_name_age(df_indic['seriesName'], df_indic['Age'], df_indic['group'])

# reorder the columns to mamke 'idx' and 'indicator_name_fmtd' the first two columns
df_indic = df_indic[['idx', 'indicator_name_fmtd'] + list(df_indic.columns[:-2])]

df_indic.to_csv('data/ppi/pipeline_indicators_normalized_2013_2016.csv', index=False)