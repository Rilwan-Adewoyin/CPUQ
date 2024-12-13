# Create b2i edges candidates for the CPUQ and verbalization
# Due to compute restraints we basically use our LLM to filter the candidate suggestions based on the naive methodology -> naive_b2i_finegrained.csv

import os
import pandas as pd
import regex as re

# Function to create intuitive concatenation of indicator name and age
def concatenate_name_age(name, age, group):

    # Check if the name string as any form of the age string in it and if not, add it
    if age in name:
        pass
    elif re.search(r'\d+', age) and re.search(r'\d+', age).group() in name:
        # check if any number extracted from the age string is in the name string
        pass
    # elif re.search(r'\d+[- ]\d?', age) and re.search(r'\d+[- ]\d?', age).group() in name:
    #     # check if there is an age / age range in the name string already
    #     pass
    elif re.search(r'\d+', name) is not None:
        # check if there is any age in the string already
        pass
    else:
        name = f"{name} ({age})"         


    # Adding information on the decile
    if 'Most deprived decile' in group:
        name = name + ' for the most deprived decile of the population.'
    elif 'Least deprived decile' in group:
        name = name + ' for the least deprived decile of the population.'
    else:
        pass
         
    # remove double spaces
    name = name.replace("  ", " ")
    return name

# Reading in indicator data
df_indicators = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_sample_raw.csv'), usecols=['seriesCode','seriesName','Age','group']).rename({'seriesCode':'indicator_code', 'seriesName':'indicator'}, axis=1)

# Reading in fine grained budget item data
df_finegrained_budget_items = pd.read_csv(os.path.join('data','ppi','data_expenditure_raw.csv'), usecols=['seriesCode','seriesName', 'category']).rename({'seriesName':'budget_item_finegrained', 'seriesCode':'bi_fg_code'}, axis=1)


# In category column replace all instances of 'Env & Reg' with 'Environmental Policy and Regulation
# In category column replace all instance sof 'Central' with 'Central Services'
df_finegrained_budget_items['category'] = df_finegrained_budget_items['category'].replace('Env & Reg', 'Environmental Policy and Regulation')
df_finegrained_budget_items['category'] = df_finegrained_budget_items['category'].replace('Central', 'Central Services')
df_finegrained_budget_items['categorgy'] = df_finegrained_budget_items['category'].replace('Highways', 'Highways and Transport services')
df_finegrained_budget_items['category'] = df_finegrained_budget_items['category'].replace('Planning', 'Planning and Development services')

# Reading in the naive b2i fine grained budget item data
# We only keep candidates that appear in this naive b2i fine grained budget item data
# We use the indicator code and bi_fg_code columns to filter potential candidates
df_naive_b2i_finegrained = pd.read_csv(os.path.join('data','ppi', 'b2i_networks','naive_b2i_finegrained.csv'), usecols=['indicator_code','bi_fg_code'] )
li_icode_bifgcode = df_naive_b2i_finegrained.values.tolist()

candidates = []
for i in range(len(df_finegrained_budget_items)):
    
    budget_item_code = df_finegrained_budget_items.iloc[i]['bi_fg_code']
    budget_item_name = df_finegrained_budget_items.iloc[i]['budget_item_finegrained']
    
    # removing the text 'Spend: ' from the budget item name
    budget_item_name_fmtd = budget_item_name.replace("Spend: ", "")

    for j in range(len(df_indicators)):
    
        indicator_code = df_indicators.iloc[j]['indicator_code']

        # skip if the indicator code and budget item code combination is not in the naive b2i fine grained budget item data
        if [indicator_code, budget_item_code] not in li_icode_bifgcode:
            continue

        indicator_name = df_indicators.iloc[j]['indicator']
        indicator_age = df_indicators.iloc[j]['Age']
        indicator_group = df_indicators.iloc[j]['group']

        indicator_name_fmtd = concatenate_name_age(indicator_name, indicator_age, indicator_group)

        candidate = {
            "budget_item_code": budget_item_code,
            "indicator_code": indicator_code,
            
            "budget_item": budget_item_name_fmtd,
            "indicator": indicator_name_fmtd,
            
            "budget_item_original_name": budget_item_name,
            "indicator_original_name": indicator_name,
            "indicator_age": indicator_age,
            "indicator_group": indicator_group
        }

        candidates.append(candidate)
    

# Make dataframe from list of dictionaries
df_candidates = pd.DataFrame(candidates)

# filter b2i_df_candidates for budget items that are in the pipeline
df_pipeline_finegrained_budget_items = pd.read_csv(os.path.join('data','ppi','pipeline_expenditure_finegrained.csv', usecols=['seriesName', 'category'])).rename({'seriesName':'budget_item_finegrained'}, axis=1)
df_candidates = df_candidates[df_candidates['budget_item'].isin(df_pipeline_finegrained_budget_items['budget_item_finegrained'])]

# Save to csv
df_candidates.to_csv(os.path.join('data','ppi','b2i_networks',"b2i_candidates.csv"), index=False)

