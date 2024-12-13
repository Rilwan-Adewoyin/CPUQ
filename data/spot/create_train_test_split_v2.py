import pandas as pd
import numpy as np
from prompt_engineering.utils_prompteng import create_negative_examples_b2i
from sklearn.model_selection import train_test_split
"""
This file is used to preprocess the spot data in an improved manner.

    # First we create two sets of datasets
    #     - Set 1)
    #         - spot_b2i_broad.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, budget item, type 
    #             - This excludes budget_items that are sub-categories of broader budget_items
    #         - From this we create a train and test set, that include negative examples
    #             - spot_b2i_broad_train.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, indicator, label
    #             - spot_b2i_broad_test.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, indicator, label

    #     - Set 2)
    #         - spot_b2i_fine.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, budget item, type
    #             - This includes budget_items that are sub-categories of broader budget_items and excludes the broader budget items that have no sub-categories
    #             - If there is a "Total ..." spend budget item and other budget_items, we remove it and replace it with "Total .. excluding ...{other_spend_budget_items}" 
    #         - From this we create a train and test set, that include negative examples
    #             - spot_b2i_fine_train.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, indicator, label
    #             - spot_b2i_fine_test.csv : Dataset containing rows with 4 columns:  budget_item_id, budget_item, indicator, label


    #     - Notes on budget item name formatting)
    #         - In our output we extract each row that corresponds that has a type of 'Outcome' - e.g. effect of govt spending on 'budget_item'
    #         - The rows with type = 'Spend', have 'indicator' columns which usually provide more detail on the type of 'budget_item' - e.g. type=Spend, 'budget_item' = Child Health  indicator:  'Spend: Children 5-19 public health programmes'
    #         - For Set 1)
    #               - Essentially we amalgamate the descriptions of the 'Spend' rows that have the same budget item
    #                   - For a budget_item if there is only one line with type == 'Spend' we use the name for this item as the budget_item name
    #                   - For a budget_item if there is more than one line with type == 'Spend':
    #                       - If one line has 'Total ' in it then we use that line as the budget_item name
    #                       - then concatenate or just use the original budget item name
    #               - Therefore we join the 'indicator' column from the 'Spend' rows to the 'Outcome' rows, and use this as the 'budget_item_name' column in the output
    #               - Note: Sometimes there are multiple type='Spend' rows per budget_item. In that case we do not perform this swap and just keep the original 'budget_item' column as the 'budget_item_name' column
    #               - The budget_item column is renamed to 'budget_category' to avoid confusion
    #         - For Set 2)
    #               - Each budget_item 'Spend' row is kept as a seperate budget item
    #               - If there is a "Total ..." spend budget item and other budget_items, we remove it and replace it with "Total .. excluding ...{other_spend_budget_items}"
    #     - Notes on dealing with 'Public Health' budget_item
    #         - The other forms of health e.g. Sexual Health, Mental Health are subsets of Public Health
    #         - In the dataset budget items will be tied to the broad 'Public Health' budget_item and also to the more specific broad budget_item e.g. 'Sexual Health'
    #         - Therefore we remove the 'Public Health' budget_item from the dataset
"""

random_state_seed = 10

def create_b2i_broad( spot_indicator_mapping_table ):
   
    # Create a subdataframe of spot_indicator_mapping_table for all rows where type is 'Spend'
    spot_indicator_mapping_table_spend = spot_indicator_mapping_table[
        spot_indicator_mapping_table['type'] == 'Spend'].copy()
    spot_indicator_mapping_table_outcome = spot_indicator_mapping_table[
        spot_indicator_mapping_table['type'] == 'Outcome'].copy()

    # Create a list of dictionary which will contain the mapping from budget_item to 
    def create_map_to_new_budget_item_name( spot_indicator_mapping_table_spend: pd.DataFrame ):
        # create a grouped pd.DataFrame  grouped on the budget_item column and including the indicator column
        grouped = spot_indicator_mapping_table_spend.groupby( 'budget_item' )['indicator'].apply( list ).reset_index( name='indicator' )

        # Convert to a dictionary where the key is the budget_item and the value is the list of indicators
        grouped_dict = grouped.set_index( 'budget_item' ).T.to_dict( 'list' )

        # Implement the logic to decide the new budget_item_name based on the list of indicator strings
        for key, li_indicators in grouped_dict.items():
            li_indicators = li_indicators[ 0 ]
            # If there is only one indicator string, then use that as the new budget_item_name
            if len( li_indicators ) == 1:
                grouped_dict[ key ] = li_indicators[ 0 ].replace( 'Total ', '')
            # If there is more than one indicator string, then use the one with 'Total' in it as the new budget_item_name
            elif len( li_indicators ) > 1 and any( 'Total' in s for s in li_indicators ):
                grouped_dict[ key ] = next( s for s in li_indicators if 'Total ' in s ).replace( 'Total ', '' )
            # If there is more than one indicator string, and none of them have 'Total' in it, then concatenate the strings
            else:
                grouped_dict[ key ] = ' & '.join( li_indicators )
        
        map_update_budget_item_name = grouped_dict

        return map_update_budget_item_name

    map_update_budget_item_name = create_map_to_new_budget_item_name( spot_indicator_mapping_table_spend )

    # Convert the old budget_item column to budget_item_original
    spot_indicator_mapping_table_outcome['budget_item_original'] = spot_indicator_mapping_table_outcome['budget_item']

    # Update the budget_item column with the new budget_item names
    spot_indicator_mapping_table_outcome['budget_item'] = spot_indicator_mapping_table_outcome['budget_item_original'].map( map_update_budget_item_name )

    return spot_indicator_mapping_table_outcome

def create_b2i_finegrained(spot_indicator_mapping_table):
    
    # Create a subdataframe of spot_indicator_mapping_table for all rows where type is 'Spend'
    spot_indicator_mapping_table_spend = spot_indicator_mapping_table[
        spot_indicator_mapping_table['type'] == 'Spend'].copy()
    spot_indicator_mapping_table_outcome = spot_indicator_mapping_table[
        spot_indicator_mapping_table['type'] == 'Outcome'].copy()



    # Create a dictionary which will contain the mapping from budget_item to a list of new budget_item names to create e.g. 1 to many mapping
    def create_map_to_new_budget_item_names( spot_indicator_mapping_table_spend: pd.DataFrame ):
        # create a grouped pd.DataFrame  grouped on the budget_item column and including the indicator column
        grouped = spot_indicator_mapping_table_spend.groupby( 'budget_item' )['indicator'].apply( list ).reset_index( name='indicator' )

        # Convert to a dictionary where the key is the budget_item and the value is the list of indicators
        grouped_dict = grouped.set_index( 'budget_item' ).T.to_dict( 'list' )

        # Implement the logic to decide the new budget_item_name based on the list of indicator strings
        for key, li_indicators in grouped_dict.items():
            # If there is only one indicator string, then use that as the new budget_item_name
            if len( li_indicators ) == 1:
                grouped_dict[ key ] = li_indicators

            # If there is a "Total ..." spend budget item and other budget_items, we remove it and replace it with "Total .. excluding ...{other_spend_budget_items}"
            elif len( li_indicators ) > 1 and any( 'Total' in s for s in li_indicators ):
                
                total_key =  next( s for s in li_indicators if 'Total ' in s )
                other_keys = [ s for s in li_indicators if 'Total ' not in s ]

                total_key_excl_others = total_key + ' (excluding ' + ', '.join( other_keys ) + ')'

                new_keys = [ total_key_excl_others ] + other_keys
                grouped_dict[ key ] = new_keys

            # If there is more than one indicator string, and none of them have 'Total' in it, then concatenate the strings
            else:
                grouped_dict[ key ] = li_indicators
        
        map_update_budget_item_name = grouped_dict

        return map_update_budget_item_name

    map_update_budget_item_name = create_map_to_new_budget_item_names( spot_indicator_mapping_table_spend )

    # Convert the map_update_budget_item_name dictionary to a dataframe with two columns: budget_item_original and budget_item, where the key is the budget_item_original and the value is the budget_item. Each item in the value list should be a seperate row
    df_map_update_budget_item_name = pd.DataFrame.from_dict( map_update_budget_item_name, orient='index' ).reset_index()
    df_map_update_budget_item_name.columns = ['budget_item_original', 'budget_item']
    df_map_update_budget_item_name = df_map_update_budget_item_name.explode( 'budget_item' )

    # Convert the budget_item column to budget_item_original    
    spot_indicator_mapping_table_outcome['budget_item_original'] = spot_indicator_mapping_table_outcome['budget_item']
    # delete budget_item column
    del spot_indicator_mapping_table_outcome['budget_item'] 

    # Perform an inner join on the spot_indicator_mapping_table_outcome and df_map_update_budget_item_name dataframes
    spot_indicator_mapping_table_outcome = spot_indicator_mapping_table_outcome.merge( df_map_update_budget_item_name, on='budget_item_original', how='inner' )

    return spot_indicator_mapping_table_outcome

def create_train_test_split(spot_indicator_mapping_table_outcome, random_state_seed=10):
    

    """
        This script creates the train and test splits for the SPOT dataset.
    """

    # Creating target field
    spot_indicator_mapping_table_outcome['related'] = 'Yes'

    # Create negative examples
    random_state = np.random.RandomState(random_state_seed)

    # Replace budget_item with 'Central' with 'Central Services'
    spot_indicator_mapping_table_outcome['budget_item'] = spot_indicator_mapping_table_outcome['budget_item'].replace('Central', 'Central Services')

    # create negative examples
    spot_indicator_mapping_table_outcome = create_negative_examples_b2i(spot_indicator_mapping_table_outcome, random_state=random_state )

    # Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
    spot_indicator_mapping_table_outcome = spot_indicator_mapping_table_outcome.groupby(['budget_item','related']).filter(lambda x: len(x) > 1)

    # perform stratified split of a dataframe into train and test subsets
    train, test = train_test_split(spot_indicator_mapping_table_outcome, test_size=0.8, random_state=random_state, stratify=spot_indicator_mapping_table_outcome[['budget_item','related']])
    return train, test

# Read the spot_indicator_mapping_table.csv file into a pandas dataframe
spot_indicator_mapping_table = pd.read_csv(
    'data/spot/spot_indicator_mapping_table_v2.csv')

# Removing any qoutation marks surrounding the text in columns: budget_item or indicator
spot_indicator_mapping_table['indicator'] = spot_indicator_mapping_table['indicator'].str.replace( 'Spend: ', '' )

spot_indicator_mapping_table['budget_item'] = spot_indicator_mapping_table['budget_item'].str.replace(
    '"', '')
spot_indicator_mapping_table['indicator'] = spot_indicator_mapping_table['indicator'].str.replace(
    '"', '')

# Remove any rows with 'Public Health' in the budget_item column
spot_indicator_mapping_table = spot_indicator_mapping_table[spot_indicator_mapping_table['budget_item'] != 'Public Health']


# Create Broad and Finegrained versions of the spot_indicator_mapping_table
spot_indicator_mapping_table_outcome_broad = create_b2i_broad( spot_indicator_mapping_table)

spot_indicator_mapping_table_outcome_finegrained = create_b2i_finegrained( spot_indicator_mapping_table )


# Then do the train_test_split for broad
spot_indicator_mapping_table_outcome_broad_train, spot_indicator_mapping_table_outcome_broad_test = create_train_test_split( spot_indicator_mapping_table_outcome_broad, random_state_seed )

# Saving Files

# Drop 'type' column from all dataframes
spot_indicator_mapping_table_outcome_broad.drop( 'type', axis=1, inplace=True )
spot_indicator_mapping_table_outcome_broad_train.drop( 'type', axis=1, inplace=True )
spot_indicator_mapping_table_outcome_broad_test.drop( 'type', axis=1, inplace=True )
spot_indicator_mapping_table_outcome_finegrained.drop( 'type', axis=1, inplace=True )



# Reorder columns to 'budget_item_original', 'budget_item', 'id', 'indicator', 'related' 
spot_indicator_mapping_table_outcome_broad = spot_indicator_mapping_table_outcome_broad[['budget_item_original', 'budget_item', 'id', 'indicator', 'related']]
spot_indicator_mapping_table_outcome_broad_train = spot_indicator_mapping_table_outcome_broad_train[['budget_item_original', 'budget_item', 'id', 'indicator', 'related']]
spot_indicator_mapping_table_outcome_broad_test = spot_indicator_mapping_table_outcome_broad_test[['budget_item_original', 'budget_item', 'id', 'indicator', 'related']]
spot_indicator_mapping_table_outcome_finegrained = spot_indicator_mapping_table_outcome_finegrained[['budget_item_original', 'budget_item', 'id', 'indicator']]

spot_indicator_mapping_table_outcome_broad.to_csv( 'data/spot/spot_b2i_broad.csv', index=False )
spot_indicator_mapping_table_outcome_broad_train.to_csv( 'data/spot/spot_b2i_broad_train.csv', index=False )
spot_indicator_mapping_table_outcome_broad_test.to_csv( 'data/spot/spot_b2i_broad_test.csv', index=False )
spot_indicator_mapping_table_outcome_finegrained.to_csv( 'data/spot/spot_b2i_finegrained.csv', index=False )

