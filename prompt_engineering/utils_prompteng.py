import pandas as pd
import math
import torch
from torch.nn import CrossEntropyLoss


import numpy as np
from contextlib import nullcontext
import random
import copy
from random import sample 
from functools import reduce
import operator
import random
from typing import Optional
import langchain
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from transformers import PreTrainedModel
from time import sleep
from itertools import islice
import peft
from peft import PeftModel, PeftModelForCausalLM
from functools import lru_cache
from transformers import AutoTokenizer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import langchain_community
import langchain_core
map_relationship_promptsmap ={}

# region budgetitem to indicator templates
li_prompts_yes_no_question_b2i = [    
    # f"Yes or No, does government spending on {{budget_item}} {{effect_type}} affect the level of {{indicator}}?"
    {
        "user_message":f"Yes or No, does government spending on {{budget_item}} {{effect_type}} affect the level of {{indicator}}?",
        "assistant_message":None
    }
]

li_prompts_openended_question_b2i = [    
    # 'Does government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',
    {
        "user_message":'Does government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',
        "assistant_message":None
    }
]

li_prompts_reasoning_question_b2i = [
    # 'To what extent, if any, does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?'
    {
        "user_message":'To what extent, if any, does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',
        "assistant_message":None
    }
]

# region Prompts for the open prompt style methodology with categorical parse style
map_category_answer_b2i = { '1':'Local government spending on "{budget_item}" does {effect_type} affect "{indicator}".', 
                                        '2':'Local government spending on "{budget_item}" does not {effect_type} affect "{indicator}"' }
map_category_label_b2i = {'1':'Yes',
                       '2':'No'}

li_prompts_categorical_question_w_reasoning_b2i: list[dict] = [ 
    {
        "user_message":f'Write only the number of the category that best fits the following statement.\nStatement: {{statement}}\nCategories:1) {map_category_answer_b2i["1"]}\t2) {map_category_answer_b2i["2"]}',
        "assistant_message":"The correct number is "
    }
]

li_prompts_categorical_question_w_reasoning_reversed_b2i: list[dict] = [
        # f'The Statement below expresses an opinion on whether local government spending on "{{budget_item}}" {{effect_type}} affects "{{indicator}}". Classify the statement\'s opinion using one of the following categories and respond only with the category number: 1) {map_category_answer_b2i["2"]}, 2) {map_category_answer_b2i["1"]}.\nStatement: {"{statement}"}'
    # f'Statement: {"{statement}"}\n\nCategories:\n1) {map_category_answer_b2i_b2i_b2i_b2i_b2i_b2i["2"]}\t2) {map_category_answer_b2i["1"]}\n\nWrite the number of the category that fits the statement'
    # f'Statement: {"{statement}"}\n\nCategories:\n1) {map_category_answer_b2i["2"]}\n2) {map_category_answer_b2i["1"]}\n\nWrite the number of the category that fits the statement'
    # f'Write "1" if the following statement implies {map_category_answer_b2i["2"]} or write "2" if it implies {map_category_answer_b2i["1"]}.\nStatement: {"{statement}"}',
    # f'Write only the number of the category that fits the following statement.\nStatement: {{statement}}\nCategories:\t1) {map_category_answer_b2i["2"]}\t2) {map_category_label_b2i["1"]}',
    {
        "user_message":f'Write only the number of the category that fits the following statement.\nStatement: {{statement}}\nCategories:\t1) {map_category_answer_b2i["2"]}\t2) {map_category_label_b2i["1"]}',
        "assistant_message":"The correct number is "
    }
        ]

li_prompts_categorical_question_b2i: list[dict] = [
    {
        "user_message":f"Government spending on {{budget_item}} {{effect_type}} affects the level of {{indicator}}.\n1) Yes 2) No.",
        "assistant_message":"The correct number is "
    },

    ]

li_prompts_categorical_question_reversed_b2i: list[dict] = [
    # f'Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"? Please answer the question using one of the following categories and respond only with the number (1 or 2) of the selected category: 1) {map_category_answer_b2i["1"]}, 2) {map_category_answer_b2i["2"]}.'
    # f'Please answer the following question using one of the following categories and respond only with the number (1 or 2) of the selected category.\nCategories:\n1) {map_category_answer_b2i["2"]}\n2) {map_category_answer_b2i_b2i["1"]}. \nQuestion: Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"?'
    # f'Select the category that answers the question. Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"?\n1) {map_category_answer_b2i["1"]}\n2) {map_category_answer_b2i["2"]}'
    # f'Select the category number that answers the question. Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"?\n1) {map_category_answer_b2i["2"]}\n2) {map_category_answer_b2i_b2i_b2i["1"]}'
    # f'Write the number of the category that fits the question. Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"?\n1) {map_category_answer_b2i["2"]}\n2) {map_category_answer_b2i["1"]}'
    # f'Categories:\n1) {map_category_answer_b2i_b2i_b2i_b2i["2"]}\n2) {map_category_answer_b2i["1"]}\nWrite the number of the category that best answers whether government spending on "{{budget_item}}" {{effect_type}} affects "{{indicator}}"?'
    # f'Does government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"? Please write the number (1 or 2) of category which correctly answers the question:\nCategories:\n\t1) {map_category_answer_b2i_b2i["2"]}\n\t2) {map_category_answer_b2i["1"]}.'
    # f'Does government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"? Please write "1" if the answer is "{map_category_label_b2i["2"]}" or "2" if the answer is "{map_category_label_b2i["1"]}".'
    # f"<|im_start|>user\nGovernment spending on {{budget_item}} {{effect_type}} affects the level of {{indicator}}.\n1) No 2) Yes.<|im_end|>\n<|im_start|>assistant\n The correct number is "
    # f"<|im_start|>user\nGovernment spending on {{budget_item}} {{effect_type}} affects the level of {{indicator}}.\n1) No 2) Yes.<|im_end|>\n<|im_start|>assistant\n The correct number is ",
    {
        "user_message":f"Government spending on {{budget_item}} {{effect_type}} affects the level of {{indicator}}.\n1) No 2) Yes.",
        "assistant_message":"The correct number is "
    }
]

# endregion
budgetitem_to_indicator_prompts = {
    'li_prompts_yes_no_question':li_prompts_yes_no_question_b2i,
    'li_prompts_openended_question':li_prompts_openended_question_b2i,
    'li_prompts_reasoning_question':li_prompts_reasoning_question_b2i,
    'li_prompts_categorical_question_w_reasoning':li_prompts_categorical_question_w_reasoning_b2i,
    'li_prompts_categorical_question_w_reasoning_reversed':li_prompts_categorical_question_w_reasoning_reversed_b2i,

    'li_prompts_categorical_question':li_prompts_categorical_question_b2i,
    'li_prompts_categorical_question_reversed':li_prompts_categorical_question_reversed_b2i,

    'map_category_answer':map_category_answer_b2i,
    'map_category_label':map_category_label_b2i
}
map_relationship_promptsmap['budgetitem_to_indicator'] = budgetitem_to_indicator_prompts
#endregion

# region indicator to indicator templates
li_prompts_yes_no_question_i2i = [
    # f"Yes or No, does the level of {{budget_item}} {{effect_type}} affect the level of {{indicator}}? "
    {
        "user_message": "Yes or No, does the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?",
        "assistant_message": None
    }
    ]
li_prompts_openended_question_i2i = [
    # "Does the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?"
    {
        "user_message": "Does the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?",
        "assistant_message": None
    }
]

li_prompts_reasoning_question_i2i = [
    # "To what extent, if any, do changes in the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?"
    {
        "user_message": "To what extent, if any, do changes in the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?",
        "assistant_message": None
    }
]

# Prompts for the open prompt style methodology with categorical parse style
map_category_answer_i2i = { '1':'The level of \"{indicator1}\" does {effect_type} affect the level of \"{indicator2}\".',
                                        '2':'The level of \"{indicator1}\" does not {effect_type} affect the level of \"{indicator2}\".' }
map_category_label_i2i = {'1':'Yes',
                          '2':'No'}

li_prompts_categorical_question_w_reasoning_i2i: list[dict] = [
    # f'Write the number of the category that best fits the following statement.\nStatement: "{{statement}}"\nCategories:\n1) {map_category_answer_i2i["1"]}\t2) {map_category_answer_i2i["2"]}',

    {
        "user_message":f'Write only the number of the category that best fits the following statement.\nStatement: "{{statement}}"\nCategories:\n1) {map_category_answer_i2i["1"]}\t2) {map_category_answer_i2i["2"]}',
        "assistant_message":"The correct number is "
    }
]

li_prompts_categorical_question_w_reasoning_reversed_i2i: list[dict] = [
    # f'Write only the number of the category that best fits the following statement.\nStatement: "{{statement}}"\nCategories:\n1) {map_category_answer_i2i["2"]}\t2) {map_category_label_i2i["1"]}',
    {
        "user_message":f'Write only the number of the category that best fits the following statement.\nStatement: "{{statement}}"\nCategories:\n1) {map_category_answer_i2i["2"]}\t2) {map_category_label_i2i["1"]}',
        "assistant_message":"The correct number is "
    }
]

li_prompts_categorical_question_i2i: list[dict] = [
    # f'Write "1" if the following statement is True or "2" if it is False. The level of "{{indicator1}}" is {{effect_type}} influential to the state of "{{indicator2}}".',
    # f'I want you to write the number 1 if the following statement is true or write the number 2 if it is false. The level of "{{indicator1}}" is {{effect_type}} influential to the level of "{{indicator2}}".'
    # f'Choose 1 if the following statement is true or choose 2 if the following statement is false.\nStatement: The level of "{{indicator1}}" {{effect_type}} affects the level of "{{indicator2}}".',
    # f'Is the following statement true (1) or false (2)?\n"The level of {{indicator1}} {{effect_type}} influences the level of {{indicator2}}"',
    # f"Answer with '1' for True and '2' for False.\nTrue or False: Does the level of {{indicator1}} {{effect_type}} affect the level of {{indicator2}}?",
    # f"The level of {{indicator1}} {{effect_type}} does affect the level of {{indicator2}}.\n1) Yes 2) No",
    {
        "user_message":f"The level of {{indicator1}} {{effect_type}} does affect the level of {{indicator2}}.\n1) Yes 2) No",
        "assistant_message":"The correct number is "
    }
    
    ]

li_prompts_categorical_question_reversed_i2i: list[dict] = [
    # 'Write "2" if the following statement is True or "1" if it is False. The level of \"{indicator1}\" is {effect_type} influential to the state of \"{indicator2}\".'
    #f'I want you to write the number 1 if the following statement is false or write the number 2 if it is true. The level of "{{indicator1}}" is {{effect_type}} influential to the level of "{{indicator2}}".'
    # f'Choose 1 if the following statement is false or choose 2 if the following statement is true.\nStatement: The level of "{{indicator1}}" {{effect_type}} affects the level of "{{indicator2}}".'
    # f'Is the following statement false (1) or true (2)?\n"The level of {{indicator1}} {{effect_type}} influences the level of {{indicator2}}"'
    # f"Answer with '1' for False and '2' for True.\nTrue or False: Does the level of {{indicator1}} {{effect_type}} affect the level of {{indicator2}}?",
    # f"The level of {{indicator1}} {{effect_type}} does affect the level of {{indicator2}}.\n1) No 2) Yes",
    {
        "user_message":f"The level of {{indicator1}} {{effect_type}} does affect the level of {{indicator2}}.\n1) No 2) Yes",
        "assistant_message":"The correct number is "
    }
    ]

li_prompts_categories_scale_question_i2i: list[dict] = [
    # f'<|im_start|>user\nOn a scale of {{scale_min}} to {{scale_max}}, how strong is the effect of "{{indicator1}}" on "{{indicator2}}"?<|im_end|>\n<|im_start|>assistant\nOn a scale of {{scale_min}} to {{scale_max}} the level is a ',

    # f'<|im_start|>assistant\nOn a scale of {{scale_min}} to {{scale_max}}, the level of the effect of "{{indicator1}}" on "{{indicator2}}" is the number ',

    {
        "user_message": f'On a scale of {{scale_min}} to {{scale_max}}, how strong is the effect of "{{indicator1}}" on "{{indicator2}}"?',
        "assistant_message": f'On a scale of {{scale_min}} to {{scale_max}} the level is a '
    }

    # f'<|im_start|>user\nTell me a fact<|im_end|>\n<|im_start|>assistant\nOn a scale of {{scale_min}} to {{scale_max}}, the level of impact of "{{indicator1}}" on "{{indicator2}}" is the number ',

    # f'<|im_start|>user\nTell me a fact<|im_end|>\n<|im_start|>assistant\nOn a scale of 1 to 3, the level of impact of "{{indicator1}}" on the level of "{{indicator2}}" is approximately the number ',
    # f'<|im_start|>assistant\nOn a scale of 1 to 3, the level of impact of "{{indicator1}}" on "{{indicator2}}" is the number ',
    # f"<|im_start|>user\nThe level of {{indicator1}} has a [FILL HERE] effect on {{indicator2}}.\n1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",
    # f"<|im_start|>user\nThe level of {{indicator1}} has a [FILL HERE] effect on {{indicator2}}. Choose the correct number  1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",
    # f"<|im_start|>user\nThe level of {{indicator1}} has a [FILL HERE] effect on {{indicator2}}. Choose the correct number from the following options 1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",
    
    # f"<|im_start|>user\nWhat is the level of of effect {{indicator1}} has on {{indicator2}}? 1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",
    # f"<|im_start|>user\nRespond with the correct number for the level of effect {{indicator1}} has on {{indicator2}}? 1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",
    # f"<|im_start|>user\n{{indicator1}} has a [FILL HERE] effect on the level of {{indicator2}}.\n1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number is ",


    # f"<|im_start|>user\n{{indicator1}} has a [FILL HERE] effect on the level of {{indicator2}}.\nSelect the correct number for FILL HERE. 1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number that fills the gap is ",
    
    # "<|im_start|>user\nWhich number best fills the GAP in the following statement: {indicator1} has a [GAP] effect on {indicator2}.\n{{{{1: 'weak', 2: 'moderate', 3:'strong'}}}}<|im_end|>\n<|im_start|>assistant\nThe number is ",

    # f"<|im_start|>user\nThe level of {{indicator1}} has a [GAP] effect on the level of {{indicator2}}.\n1) weak 2) moderate 3) strong<|im_end|>\n<|im_start|>assistant\nThe number that fills the gap is",

]

# Prompts for the scaling categorisation method
category_scale_i2i = lambda s_min, s_max: list(range(s_min, s_max+1))

indicator_to_indicator_prompts = {
    'li_prompts_yes_no_question': li_prompts_yes_no_question_i2i,
    'li_prompts_openended_question': li_prompts_openended_question_i2i,
    'li_prompts_reasoning_question': li_prompts_reasoning_question_i2i,
    'li_prompts_categorical_question_w_reasoning': li_prompts_categorical_question_w_reasoning_i2i,
    'li_prompts_categorical_question_w_reasoning_reversed': li_prompts_categorical_question_w_reasoning_reversed_i2i,

    'li_prompts_categorical_question': li_prompts_categorical_question_i2i,
    'li_prompts_categorical_question_reversed': li_prompts_categorical_question_reversed_i2i,

    'li_prompts_categories_scale_question': li_prompts_categories_scale_question_i2i,

    'map_category_answer': map_category_answer_i2i,
    'map_category_label': map_category_label_i2i,

    'category_scale':category_scale_i2i
}

map_relationship_promptsmap['indicator_to_indicator'] = indicator_to_indicator_prompts
# endregion


# region SystemMessages
system_prompt_b2i_arbitrary = 'You are a socio-economic researcher tasked with answering a question about whether government spending on a "government budget item" affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'
system_prompt_b2i_directly = 'You are a socio-economic researcher tasked with answering a question about whether government spending on a "government budget item" directly affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'
system_prompt_b2i_indirectly = 'You are a socio-economic researcher tasked with answering a question about whether government spending on a "government budget item" indirectly affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'


system_prompt_i2i_arbitrary = 'You are a socio-economic researcher tasked with answering a question about whether changes in the level of a "socio-economic/health indicator" affects the level of another "socio-economic/health indicator". In the question, both of the socio-economic/health indicators will be presented within quotation marks.'
system_prompt_i2i_directly = 'You are a socio-economic researcher tasked with answering a question about whether changes in the level of a "socio-economic/health indicator" directly affects the level of another "socio-economic/health indicator". In the question, both of the socio-economic/health indicators will be presented within quotation marks.'
system_prompt_i2i_indirectly = 'You are a socio-economic researcher tasked with answering a question about whether changes in the level of a "socio-economic/health indicator" indirectly affects the level of another "socio-economic/health indicator". In the question, both of the socio-economic/health indicators will be presented within quotation marks.'

map_system_prompts_b2i = {
    'arbitrary':system_prompt_b2i_arbitrary,
    'directly':system_prompt_b2i_directly,
    'indirectly':system_prompt_b2i_indirectly,
    'yes_no':'Answer the following question with "Yes" or "No".',
    'open':'Write a conclusive, one sentence answer to the following question.',
    'categorise':'',
    'cot_categorise':'Write a thorough, detailed and conclusive four sentence answer to the following question.',
}

map_system_prompts_i2i = {
    'indirectly':system_prompt_i2i_indirectly,
    'directly':system_prompt_i2i_directly,
    'arbitrary':system_prompt_i2i_arbitrary,
    'yes_no':'Answer the following question with "Yes" or "No".',
    'open':'Write a conclusive, one sentence answer to the following question.',
    'categorise':'',
    'cot_categorise':'Write a thorough, detailed and conclusive four sentence answer to the following question.',
    'categories_scale':'Answer the following question with only a number between {scale_min} and {scale_max}.',
    'verbalize_scale':'Answer the following question with only a number.'

}

map_relationship_system_prompt = {
    'budgetitem_to_indicator':map_system_prompts_b2i,
    'indicator_to_indicator':map_system_prompts_i2i
}

system_prompt_parse_outp_categories_rules_b2i = None
system_prompt_parse_outp_categories_rules_i2i = None

map_relationship_sysprompt_categoriesanswer = {
    'budgetitem_to_indicator':system_prompt_parse_outp_categories_rules_b2i,
    'indicator_to_indicator':system_prompt_parse_outp_categories_rules_i2i
}
# endregion

# region BaseModelFormat - The format required by the underlying language model
format_vicuna_1_1 = "{system_message}\nUSER: {user_message}\nASSISTANT: "
format_vicuna_1_1_no_sysmessage = "USER: {user_message}\nASSISTANT: "

format_alpaca = "{system_message}\n\n### Instruction:\n{user_message}\n\n### Response:\n"
format_alpaca_no_sysmessage = "### Instruction:\n{user_message}\n\n### Response:\n"

format_beluga = "### System:\n{system_message}\n\n### User:\n{user_message}\n\n### Assistant:\n"
format_beluga_no_sysmessage = "### User: {user_message}\n\n### Assistant:\n"

format_dummy = "{system_message}\n\n{user_message}\n\n"
format_dummy_no_sysmessage = "{user_message}\n\n"

def extract_assistant_answer(text:str, llm_name:str) -> str:

    if "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" in text:
        return text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[-1].strip()

    elif any(x in llm_name for x in ['hermes-llama-2','hermes-llama2','yi']):
        return text.split('ASSISTANT: ')[-1].strip()

    elif any(x in llm_name for x in ['vicuna', 'lazarus', 'minotaur']):
        return text.split('Response:\n')[-1].strip()

    elif any(x in llm_name for x in ['beluga','llama-2-70b-instruct-v2', 'llama-30b-instruct-2048', 'llama-3.2-1b-instruct', 'llama-3.2-3b-instruct', 'llama-3.1-8b-instruct', 'llama-3.1-70b-instruct','meta-llama']):

        if 'Assistant' in text:
            return text.split('Assistant')[-1].rstrip('\n').strip()
        elif 'assistant' in text:
            return text.split('assistant')[-1].rstrip('\n').strip()
        else:
            return text.strip()
    
   

    else:
        raise ValueError(f'Unknown llm_name: {llm_name}')

def map_llmname_input_format(llm_name, user_message, system_message=None, response=None):

    assert user_message is not None
    if system_message is not None:
        system_message = system_message.strip(' ')

    llm_name = llm_name.lower()
    
    if any(x in llm_name for x in ['hermes-llama-2','hermes-llama2','yi','alpaca', 'guanaco']):
        if system_message is not None:
            template = format_alpaca
        else:
            template = format_alpaca_no_sysmessage
    
    elif any(x in llm_name for x in ['vicuna', 'lazarus', 'minotaur']):
        if system_message is not None:
            template = format_vicuna_1_1
        else:
            template = format_vicuna_1_1_no_sysmessage
    
    elif any( x in llm_name for x in ['beluga','llama-2-70b-instruct-v2', 'llama-30b-instruct-2048', 'llama-3.2-1b-instruct', 'llama-3.2-3b-instruct', 'llama-3.1-8b-instruct', 'llama-3.1-70b-instruct']):
        if system_message is not None:
            template = format_beluga
        else:
            template = format_beluga_no_sysmessage
    
    elif any(x in llm_name for x in ['dummy', 'test']) :
        if system_message is not None:
            template = format_dummy
        else:
            template = format_dummy_no_sysmessage


    else:
        raise ValueError(f'Unknown llm_name: {llm_name}')

    if system_message is not None:
        template = template.format(system_message=system_message, user_message=user_message)
    else:
        template = template.format(user_message=user_message)

    # Adding response
    if response is not None:
        template += response

    return template
# endregion

def create_negative_examples_b2i(dset:pd.DataFrame, random_state=None) -> pd.DataFrame:
    """Create negative examples of budget items that affect indicators
         for the Spot Dataset by randomly selecting a budget item and indicator
        from the dataset and then swapping them
    
    dset: pd.DataFrame
        The dataset to create negative examples from
    
    Returns
    -------
    pd.DataFrame
        The dataset with negative
    
    """
    l = len(dset)
    # Each budget_item has n records
    # For each budget_item we sample min(n,l-n) false examples
    # These false examples are created by filtering on other budget_items and then sampling
    
    for budget_item in dset['budget_item'].unique():
        dset_budget_item = dset[dset['budget_item']==budget_item]
        n = len(dset_budget_item)
        

        dset_budget_item_neg = dset[dset['budget_item']!=budget_item].sample(min(n,l-n), replace=False, random_state=random_state) 
        
        dset_budget_item_neg['budget_item'] = budget_item
        dset_budget_item_neg['related'] = 'No'
        dset_budget_item_neg['budget_item_original'] = dset_budget_item['budget_item_original'].iloc[0]
        
        dset = pd.concat([dset, dset_budget_item_neg], axis=0)

    return dset

def joint_probabilities_for_category(
    li_text, model, tokenizer, batch_size: int = 1, 
    max_length=None, category_token_len=1,
    map_tokenidx_similartokens:dict[int,list[int]]=None):


    """For a given prompt taking the style of "Answer with the letter of the Category which best answers my question", This function returns the joint probabilities for the category tokens in each posible answer,
        NOTE: by design the category responses must all be the same length, ideally 1 token length.

        NOTE: However the function is currently written to work on sequencs longer than 1 token, but this is not recommended.
    """

    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from peft import PeftModel
    from torch.nn.functional import log_softmax

    assert isinstance(model, PreTrainedModel) or isinstance(model, PeftModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert category_token_len == 1, "Currently only supports category tokens of length 1"

    model = model
    tokenizer = tokenizer

    if tokenizer.pad_token is None:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
        # tokenizer.pad_token = existing_special_tokens[0]

    joint_probs = []

    for start_index in range(0, len(li_text), batch_size):
        end_index = min(start_index + batch_size, len(li_text))

        # Encode the data for this sub batch
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        encodings = tokenizer(
            li_text[start_index:end_index],
            add_special_tokens=True,
            padding=True,
            truncation=False,
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        encoded_texts_batch = encodings["input_ids"]
        attn_masks_batch = encodings["attention_mask"]
        
        labels = encoded_texts_batch

        with torch.no_grad():
            out_logits = model(encoded_texts_batch, attention_mask=attn_masks_batch).logits

        shift_logits = out_logits[..., :-1, :]
        shift_labels = labels[..., 1: ]
        shift_attention_mask_batch = attn_masks_batch[..., 1:]

        shift_logits = shift_logits[..., -category_token_len:, :]
        shift_labels = shift_labels[..., -category_token_len:]
        shift_attention_mask_batch = shift_attention_mask_batch[..., -category_token_len:]

        shift_logits = shift_logits.contiguous()
        shift_labels = shift_labels.contiguous()
        shift_attention_mask_batch = shift_attention_mask_batch.contiguous()

        # Calculate probabilities from logits
        log_probs  = log_softmax(shift_logits, dim=-1)

        # Use gather to select the log probabilities for the actual tokens

        if category_token_len == 1 and map_tokenidx_similartokens is not None:
            for shift_label in shift_labels.split(1):
                # Get the aggregated log probabilities for the token shift_label and all similar tokens
                gathered_log_probs = log_probs.gather(-1, shift_label.unsqueeze(-1)).squeeze(-1)
                # the shift_labels shoule be a single value, so we can use it to get the similar tokens
                similar_tokens = map_tokenidx_similartokens[shift_label.item()]
                gathered_log_probs_similar = log_probs.gather(-1, torch.tensor(similar_tokens)[None,None,:].to(log_probs.device)).squeeze(-1)
                
                probs = torch.exp(gathered_log_probs).flatten() + torch.exp(gathered_log_probs_similar).sum()
                # probs_agg = probs.sum(dim=-1)
                joint_probs += probs.tolist()
            
        
        else:
            gathered_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            gathered_log_probs = gathered_log_probs * shift_attention_mask_batch
            # Sum the log probabilities for the actual tokens to get the joint log probability
            joint_log_prob_batch = gathered_log_probs.sum(dim=-1)
            joint_prob_batch = torch.exp(joint_log_prob_batch)

            joint_probs += joint_prob_batch.tolist()

    return joint_probs

def nomalized_probabilities( probs: dict[str,float]) -> dict[str,float]:
    
        """Normalises a dictionary of probabilities"""
        # Normalise probabilities
        norm_probs = {}
        total = sum(probs.values())
        for k,v in probs.items():
            norm_probs[k] = v/total
    
        return norm_probs

class PromptBuilder():
    def __init__(self, 
                 llm,
                 llm_name,
                 prompt_style:str,
                 k_shot:int,
                 ensemble_size:int, 
                 examples_dset:list[dict]|None=None, 
                 effect_type:str="arb'itrary", 
                 relationship:str="budgetitem_to_indicator",
                 seed:int=10,
                 **kwargs ) -> None:
        """
            unbias_categorisations (bool): If true, the PromptBuilder builds two categorical answer prompts with the order of categories reversed in order to remove any bias towards select 1st of 2nd answer  
        """
        
        assert prompt_style in ['yes_no','open', 'categorise', 'cot_categorise', 'categories_scale', 'verbalize_scale' ], "Prompt style must be either yes_no, open, categorise or cot_categorise"
        assert effect_type in ['arbitrary', 'directly', 'indirectly'], "Effect order must be either arbitrary, directly or indirectly"
        assert relationship in ['budgetitem_to_indicator', 'indicator_to_indicator'], "Relationship must be either budgetitem_to_indicator or indicator_to_indicator"
        assert k_shot <= len(examples_dset) if examples_dset is not None else True, "User can not create a K-shot context with more examples than the number of examples in the dataset"

        if prompt_style == 'cot_categorise':
            assert k_shot == 0, "K-shot must be 0 for cot prompts"

        self.llm = llm

        self.llm_name = llm_name
        self.tokenizer = kwargs.get('tokenizer', None)
        self.prompt_style = prompt_style
        self.k_shot = k_shot    # Number of examples to use as context for each prompt
        self.ensemble_size = ensemble_size # Number of different prompts to use per prediction
        self.examples_dset = examples_dset
        self.effect_type = effect_type
        self.relationship = relationship
        self.seed = seed
        random.seed(seed)
        # when arbitrary is subbed into the prompt template, it will result in a double space in the prompt. We use .replace("  ", " ") to remove this

    
    def effect_type_str(self) -> str:
        return '' if self.effect_type == 'arbitrary' else self.effect_type 
    
    def __call__(self, batch:list[dict], reverse_categories_order:Optional[bool]=False, **kwargs) -> list[list[str]]:
        """
            Given a batch of examples, this function returns a list of prompts for each example in the batch
        """

        # First we generate an ensemble of templates to be filled in for each element in the batch
        if self.prompt_style == 'yes_no':
            templates = self._yes_no_template()
        elif self.prompt_style == 'open':
            templates = self._open_template()
        elif self.prompt_style == 'categorise':
            # To ensure both the templates selected are the same for each example in the batch, we use the seed to ensure the random selection is the same for each example
            templates = self._categorise_template(reverse_categories_order=reverse_categories_order, seed=self.seed)
        elif self.prompt_style == 'cot_categorise':
            templates = self._cot_template(seed=self.seed)
        elif self.prompt_style == 'categories_scale':
            scale_min = kwargs.get('scale_min', 1)
            scale_max = kwargs.get('scale_max', 5)
            templates = self._categories_scale_template(scale_max=scale_max, scale_min=scale_min)
        elif self.prompt_style == 'verbalize_scale':
            scale_min = kwargs.get('scale_min', 1)
            scale_max = kwargs.get('scale_max', 5)
            templates = self._categories_scale_template(scale_max=scale_max, scale_min=scale_min)

        else:
            raise ValueError('Invalid prompt_style: ' + self.prompt_style)
    
        # Second given a k_shot prompt template, we then create n = ensemble_size, realisations of the template by sampling from the training set
        if self.prompt_style == 'yes_no':
            li_filled_templates, li_li_discourse = self.fill_template_yesno(templates, batch, **kwargs)
        elif self.prompt_style == 'open':
            li_filled_templates, li_li_discourse = self.fill_template_open(templates, batch)
        elif self.prompt_style == 'categorise':
            li_filled_templates, li_li_discourse = self.fill_template_categorise(templates, batch, reverse_categories_order=reverse_categories_order)
        elif self.prompt_style == 'cot_categorise':
            li_filled_templates, li_li_discourse = self.fill_template_cot(templates, batch, reverse_categories_order=reverse_categories_order)
        elif self.prompt_style == 'categories_scale':
            li_filled_templates, li_li_discourse = self._fill_template_categories_scale(templates, batch)
        elif self.prompt_style == 'verbalize_scale':
            li_filled_templates, li_li_discourse = self._fill_template_verbalize_scale(templates, batch, **kwargs)

        else:
            li_filled_templates = []
            li_li_discourse = []
         
        return li_filled_templates, li_li_discourse
    
    @lru_cache(maxsize=2)
    def get_generation_params(self, prompt_style:str, **gen_kwargs):
        generation_params = {}
        _ = {
            langchain.llms.huggingface_pipeline.HuggingFacePipeline:'max_new_tokens',
            langchain_community.chat_models.ChatOpenAI:'max_tokens',
            PeftModel:'max_new_tokens',
            PreTrainedModel:'max_new_tokens'
        }
        k = _.get( next( (k for k in _.keys() if isinstance(self.llm, k)) ), None  )        
        
        
        if prompt_style == 'yes_no':
            generation_params[k] = 5
        elif prompt_style == 'open':
            generation_params[k] = 200
        elif prompt_style == 'categorise':
            generation_params[k] = 2
        elif prompt_style == 'cot_categorise':
            generation_params[k] = 300
        elif prompt_style == 'verbalize_scale':
            generation_params[k] = 2
        
        if isinstance(self.llm, langchain.llms.huggingface_pipeline.HuggingFacePipeline ):
            generation_params['early_stopping'] = True
            generation_params['do_sample'] = False

        if isinstance(self.llm, PeftModel ):
            generation_params['do_sample'] = False
        
        if isinstance(self.llm, PreTrainedModel ):
            generation_params['do_sample'] = False
            generation_params['early_stopping'] = True

        
        # Overriding any set keys with k,v in gen_kwargs
        for k,v in gen_kwargs.items():
            if k in generation_params:
                generation_params[k] = v
            
        return generation_params

    def generate(self, li_li_prompts, include_user_message_pre_prompt=True, include_system_message=True, add_suffix_space=False,**gen_kwargs):
        li_li_preds= []
        li_li_prompts_fmtd = []

        # Shift away from HuggingFace pipeline to allow for batched processing
        
        if gen_kwargs.get('gpu_batch_size',1) >1 and isinstance(self.llm, HuggingFacePipeline):
            # tokenizer = self.llm.pipeline.tokenizer
            model = self.llm.pipeline.model
            
        else:
            model = self.llm

        if isinstance(model, langchain_community.chat_models.ChatOpenAI): #type: ignore
            
            generation_params = self.get_generation_params(self.prompt_style, **gen_kwargs)
            
            for k,v in generation_params.items():
                setattr(model, k, v)

            for li_prompts in li_li_prompts:
                sleep(20)
                batch_messages = [
                        [   
                            SystemMessage(content=map_relationship_system_prompt[self.relationship][self.effect_type]),
                            HumanMessage(content=map_relationship_system_prompt[self.relationship][self.prompt_style] + ' ' + prompt)
                            ] for prompt in li_prompts
                    ]
                
                
                outputs = model.generate( batch_messages )
                li_preds: list[str] = [ li_chatgen[0].text for li_chatgen in outputs.generations ]
                li_li_preds.append(li_preds)
        
        elif isinstance(model, langchain_core.language_models.llms.BaseLLM): #type: ignore

            # Set the generation kwargs - Langchain equivalent method to allow variable generation kwargs            
            for k,v in self.get_generation_params(self.prompt_style, **gen_kwargs).items():
                try:
                    model.pipeline._forward_params[k] = v
                except AttributeError:
                    model.pipeline._forward_params = {k:v}

            for li_prompts in li_li_prompts:
                
                # Formatting prompts to adhere to format required by Base Language Model
                li_prompts_fmtd = []
                
                for prompt in li_prompts:
                    if include_user_message_pre_prompt:
                        user_message = map_relationship_system_prompt[self.relationship][self.prompt_style] + ' ' + prompt["user_message"]
                    else:
                        user_message = prompt["user_message"]

                    if include_system_message:
                        system_message = map_relationship_system_prompt[self.relationship][self.effect_type]
                    else:
                        system_message = None

                    # li_prompts_fmtd.append(
                    #     map_llmname_input_format(self.llm_name,
                    #         user_message = user_message ,
                    #         system_message = system_message,

                    #         )
                    # )

                    from utils import apply_chat_templates

                    tokenizer = self.llm.pipeline.tokenizer if isinstance(self.llm, HuggingFacePipeline) else self.tokenizer    

                    prompt_fmtd = apply_chat_templates(
                            tokenizer,
                            system_message=system_message,
                            user_messages=[user_message],
                            assistant_messages=[prompt["assistant_message"]] if prompt["assistant_message"] is not None else None,
                            final_am_start_asisstant_message=True
                        )
                    

                    li_prompts_fmtd.append(prompt_fmtd)

                outputs = model.generate(
                    prompts=li_prompts_fmtd)
                
                li_preds : list[str] = [ chatgen.text.strip(' ') for chatgen in sum(outputs.generations,[]) ]
            
                # Extract just the assistant's answer
                li_preds = [ extract_assistant_answer(pred, self.llm_name.lower()) for pred in li_preds ]

                li_li_prompts_fmtd.append(li_prompts_fmtd)
                li_li_preds.append(li_preds)
        
        elif  isinstance(model, PeftModel) or isinstance(model, PreTrainedModel): #type: ignore
            
            # Setting Batch size for llm to process
            gpu_batch_size = gen_kwargs.get('gpu_batch_size', None)
            if gpu_batch_size is not None:
                # Flatten the list
                flattened = [item for sublist in li_li_prompts for item in sublist]
                
                # Backup the original shape of li_li_prompts for reshaping later
                original_shapes = [len(inner) for inner in li_li_prompts]

                # Split the flattened list into chunks of gpu_batch_size
                li_li_prompts = [flattened[i:i + gpu_batch_size] for i in range(0, len(flattened), gpu_batch_size)]


            generation_params = self.get_generation_params(self.prompt_style)

            if model.generation_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                self.tokenizer.pad_token = self.tokenizer.decode([self.tokenizer.pad_token_id])

            elif self.tokenizer.pad_token_id is None and model.generation_config.pad_token_id is not None:
                self.tokenizer.pad_token_id = model.generation_config.pad_token_id
                self.tokenizer.pad_token = self.tokenizer.decode([self.tokenizer.pad_token_id])
                
            model.generation_config.bos_token_id = self.tokenizer.bos_token_id
            model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            # model.generation_config.max_length = None

            for k,v in generation_params.items():
                setattr(model.generation_config, k, v)
            
            from utils import apply_chat_templates

            for li_prompts in li_li_prompts:
                
                li_prompts_fmtd = []
                # Formatting prompts to adhere to format required by Base Language Model                
                for prompt in li_prompts:
                    if include_user_message_pre_prompt:
                        user_message = map_relationship_system_prompt[self.relationship][self.prompt_style] + ' '+ prompt["user_message"]
                    else:
                        user_message = prompt["user_message"]

                    if include_system_message:
                        system_message = map_relationship_system_prompt[self.relationship][self.effect_type]
                    else:
                        system_message = None

                    tokenizer = self.llm.pipeline.tokenizer if isinstance(self.llm, HuggingFacePipeline) else self.tokenizer    
                    prompt_fmtd = apply_chat_templates(
                            tokenizer,
                            system_message=system_message,
                            user_messages=[user_message],
                            assistant_messages=[prompt.get("assistant_message",None)],
                            final_am_start_asisstant_message=True
                        )
                    

                    li_prompts_fmtd.append(prompt_fmtd)

                # Removing trailing space from end: (for some reason a space at the end causes to model to instaly stop generating)
                if add_suffix_space is True:
                    li_prompts_fmtd = [ prompt + ' ' for prompt in li_prompts_fmtd ]
                else:
                    li_prompts_fmtd = [ prompt.strip(' ') for prompt in li_prompts_fmtd ]

                # Decide how to pad and truncate the inputs
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(li_prompts_fmtd, return_tensors='pt', padding='longest', truncation=False ).to(model.device)
                self.tokenizer.padding_side = 'right'
                
                outputs = model.generate(**inputs, **generation_params, generation_config=model.generation_config )

                output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # li_preds = [ text[ len(prompt): ].strip(' .') for prompt, text in zip(li_prompts_fmtd, output_text) ]  

                li_preds = [ extract_assistant_answer(pred, self.llm_name.lower()) for pred in output_text ]

                li_li_prompts_fmtd.append(li_prompts_fmtd)
                li_li_preds.append(li_preds)
            
            if gpu_batch_size is not None:
                reshaped_li_prompts_fmtd = []
                reshaped_li_preds = []

                li_li_prompts_fmtd_flattened =  [item for sublist in li_li_prompts_fmtd for item in sublist] 
                li_li_preds_flattened =  [item for sublist in li_li_preds for item in sublist] 

                idx = 0
                for shape in original_shapes:
                    _1 = li_li_prompts_fmtd_flattened[idx:idx+shape]
                    _2 = li_li_preds_flattened[idx:idx+shape]

                    reshaped_li_prompts_fmtd.append( _1 )
                    reshaped_li_preds.append( _2 )
                    idx += shape

                li_li_prompts_fmtd = reshaped_li_prompts_fmtd
                li_li_preds = reshaped_li_preds

            inputs = inputs.to('cpu')    
            outputs = outputs.to('cpu')
        
        else:
            raise ValueError(f"llm type {type(model)} not recognized")
            
        return li_li_prompts_fmtd, li_li_preds

    def _yes_no_template(self) -> list[dict]:
        # This creates sets of prompt templates, one prompt set for each prediction and M different prompts within each prompt set
        # When producing a prompt set of size M<N we randomly sample 
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        
        li_prompts: list[dict] = map_relationship_promptsmap[self.relationship]['li_prompts_yes_no_question']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):
            
            # This handles the actual question excl the k_shot context
            if self.relationship == 'budgetitem_to_indicator':
                if self.k_shot >0:
                    raise NotImplementedError("Yes no template not implemented for k_shot > 0")
                    prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type_str() ).replace('  ',' ') +"\nAnswer: "
                else:

                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type_str() ).replace('  ',' '),
                        "assistant_message": templates[ens_idx]["assistant_message"]
                    }
                
            elif self.relationship == 'indicator_to_indicator':
                # prompt = "Question: "+templates[ens_idx]['user_message'].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type_str() ).replace('  ',' ') +"\nAnswer: "
                prompt = {
                    "user_message": templates[ens_idx]["user_message"].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type_str() ).replace('  ',' '),
                    "assistant_message": templates[ens_idx]["assistant_message"]
                }

            # Add k_shot context to prompt
            for k in reversed(range(self.k_shot)):
                if self.relationship == 'budgetitem_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx]['user_message'].format( budget_item=f'{{budget_item_{k}}}',  indicator=f'{{indicator_{k}}}', effect_type=self.effect_type_str() ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."
                elif self.relationship == 'indicator_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx]['user_message'].format( indicator1=f'{{indicator1_{k}}}',  indicator2=f'{{indicator2_{k}}}', effect_type=self.effect_type_str() ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."
                
                prompt["user_message"] = context_k + "\n\n"+prompt["user_message"]
            
            templates[ens_idx] = {
                "user_message": prompt['user_message'],
                "assistant_message": prompt['assistant_message']
            }

        return templates
    
    def _open_template(self) -> list[str]:
        # This creates sets of prompt templates, one prompt set for each prediction and M different prompts within each prompt set
        # When producing a prompt set of size M<N we randomly sample
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        # This output leaves gaps for budget_items, indicators, and responses ot be filled in 

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_openended_question']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )
        
        for ens_idx in range(self.ensemble_size):
            if self.relationship == 'budgetitem_to_indicator':
                if self.k_shot >0:
                    raise NotImplementedError("Open template not implemented for k_shot > 0")
                    prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type_str() ).replace('  ',' ') + "\nAnswer: "
                else:
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type_str() ).replace('  ',' '),
                        "assistant_message": templates[ens_idx]["assistant_message"]
                    }

            elif self.relationship == 'indicator_to_indicator':
                if self.k_shot >0:
                    raise NotImplementedError("Open template not implemented for k_shot > 0")
                    prompt = "Question: "+templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type_str() ).replace('  ',' ') + "\nAnswer: "
                else:
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type_str() ).replace('  ',' '),
                        "assistant_message": templates[ens_idx]["assistant_message"]
                    }
                
            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                if self.relationship == 'budgetitem_to_indicator':
                    context_k = "Question: " +templates[ens_idx]['user_message'].format( budget_item=f'{{budget_item_{k}}}', indicator=f'{{indicator_{k}}}', effect_type=self.effect_type_str() ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."
                elif self.relationship == 'indicator_to_indicator':
                    context_k = "Question: " +templates[ens_idx]['user_message'].format( indicator1=f'{{indicator1_{k}}}', indicator2=f'{{indicator2_{k}}}', effect_type=self.effect_type_str() ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."

                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = {
                "user_message": prompt,
                "assistant_message": templates[ens_idx]["assistant_message"]
            }

        return templates

    def _categorise_template(self, reverse_categories_order=False, seed=None)  -> list[str]:
        """
            If reverse_categories_order is True, then the categories are reversed. Used in order to remove bias in the model towards the first category.
            seed sets the seed controlling the order of generation
        """
        if seed is not None:
            random.seed(seed)

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_categorical_question' if not reverse_categories_order else 'li_prompts_categorical_question_reversed']
        templates: list[dict] = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        assert len(templates)>=self.ensemble_size, f"len(templates)={len(templates)} < self.ensemble_size={self.ensemble_size}. Can not sample {self.ensemble_size} prompts from 'li_prompts_categorical_question' prompts."

        filled_templates = []
        for ens_idx in range(self.ensemble_size):

            if self.relationship == 'budgetitem_to_indicator':

                
                user_message = templates[ens_idx]['user_message'].format(budget_item='{target_budget_item}', indicator='{target_indicator}', effect_type=self.effect_type_str() )
                response = templates[ens_idx]['assistant_message']
   
            

            elif self.relationship == 'indicator_to_indicator':

                user_message = templates[ens_idx]['user_message'].format(indicator1='{target_indicator1}', indicator2='{target_indicator2}', effect_type=self.effect_type_str() )
                response = templates[ens_idx]['assistant_message']
            
            # filled_templates[ens_idx] = map_llmname_input_format(self.llm_name, user_message, response=response)

            filled_templates.append( {
                "user_message": user_message,
                "assistant_message": response
            } )

        return filled_templates
    
    def _cot_template(self, seed=None)  -> list[dict]:
        if seed is not None:
            random.seed(seed)

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_reasoning_question']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):

            if self.relationship == 'budgetitem_to_indicator':
                # prompt = 

                prompt = {
                    "user_message": templates[ens_idx]["user_message"].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type_str() ).replace('  ',' '),
                    "assistant_message": templates[ens_idx]["assistant_message"]
                }
            
            elif self.relationship == 'indicator_to_indicator':
                prompt = {
                    "user_message": templates[ens_idx]["user_message"].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type_str() ).replace('  ',' '),
                    "assistant_message": templates[ens_idx]["assistant_message"]
                }
                        
            templates[ens_idx] = prompt
        
        return templates

    def _categories_scale_template(self, seed=None, scale_min=1, scale_max=5) -> list[dict]:
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_categories_scale_question']
        
        # Take the first n prompts from the list of prompts
        # logging.info("Currently selec")
        
        if self.ensemble_size == -1:
            templates = copy.deepcopy(li_prompts)    
        else:
            templates = copy.deepcopy(li_prompts[:self.ensemble_size])
                
        for ens_idx in range(len(li_prompts)):
            
            if self.relationship == 'budgetitem_to_indicator':
                raise NotImplementedError("Categories scale question not implemented for budgetitem_to_indicator relationship")
            elif self.relationship == 'indicator_to_indicator':
                prompt = {
                    "user_message" :templates[ens_idx]["user_message"].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', scale_min=scale_min, scale_max=scale_max ).replace('  ',' '),
                }
                if "system_message" in templates[ens_idx]:
                    prompt["system_message"] = templates[ens_idx]["system_message"].format(scale_max=scale_max, scale_min=scale_min)

            templates[ens_idx] = prompt
        return templates

    def fill_template_yesno(self, templates:list[dict], batch:list[dict], **kwargs) -> tuple[list[list[str]], list[list[str]]]:
        """Fill in the template with the target and k_shot context"""

        li_li_prompts = []
        li_li_prompts_fmtd = []
        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            for ens_idx in range(self.ensemble_size):
                # This indented code section fills in the k_shot context with random extracts from dataset

                # sample k items from our train set into a format dict for the template
                if self.k_shot == 0:
                    format_dict = {}
                elif self.relationship == 'budgetitem_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":d['related'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} ) 
                elif self.relationship == 'indicator_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}":d['related'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} )
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    # prompt = templates[ens_idx].format(
                    #     target_budget_item= row['budget_item'].strip('.'), target_indicator=row['indicator'].strip('.'),
                    #     **format_dict
                    # )
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_budget_item= row['budget_item'].strip('.'), target_indicator=row['indicator'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message")
                         }

                elif self.relationship == 'indicator_to_indicator':
                    # prompt = templates[ens_idx].format(
                    #     target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.'),
                    #     **format_dict
                    # )
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message")
                         }
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        # Adding Prediction
        li_li_prompts_fmtd, li_li_statement = self.generate(li_li_prompts, **kwargs)
        li_li_discourse = [ [ prompt_fmtd+statement for prompt_fmtd, statement in zip(li_prompts, li_statement) ] for li_prompts, li_statement in zip(li_li_prompts_fmtd, li_li_statement) ]

        return li_li_statement, li_li_discourse

    def fill_template_open(self, templates:list[dict], batch:list[dict])->tuple[list[list[str]], list[list[str]]]:
        
        assert self.ensemble_size == 1, "Open ended questions only support ensemble_size=1, since map_category_answer_b2i is has one dictionary"
        # li_answer_templates = [  map_category_answer_b2i]
        map_category_answer = map_relationship_promptsmap[self.relationship]['map_category_answer']
        map_category_label = map_relationship_promptsmap[self.relationship]['map_category_label']

        li_answer_templates = [ map_category_answer ]

        template_responses = copy.deepcopy( sample(li_answer_templates, self.ensemble_size)  )

        li_li_prompts = []
        for row in batch:
            
            li_prompts = []
            for ens_idx in range(self.ensemble_size):
                
                ## filling k-shot examples in template
                if self.k_shot > 0:
                    # Fill in the k_shot context with random extracts from dataset
                    
                    # Sample math.ceil(k/2) positive and math.floor(k/2) negative examples
                    pos_examples_sample = random.sample( [d for d in self.examples_dset if d['related']=='Yes'], math.ceil(self.k_shot/2) )
                    neg_examples_sample = random.sample( [d for d in self.examples_dset if d['related']=='No'], math.floor(self.k_shot/2) )
                    
                    # Creating the open ended answer version of the examples
                    if self.relationship == 'budgetitem_to_indicator':
                        pos_examples_open_ended_answer = [ template_responses[ens_idx][map_category_label['Yes']].format(budget_item=d['budget_item'], indicator=d['indicator'].strip('.'), effect_type=self.effect_type_str()).replace('  ',' ') for  d in pos_examples_sample ]
                        neg_examples_open_ended_answer = [ template_responses[ens_idx][map_category_label['No']].format(budget_item=d['budget_item'], indicator=d['indicator'].strip('.'), effect_type=self.effect_type_str()).replace('  ',' ') for d in neg_examples_sample ]
                    elif self.relationship == 'indicator_to_indicator':
                        pos_examples_open_ended_answer = [ template_responses[ens_idx][map_category_label['Yes']].format(indicator1=d['indicator1'].strip('.'), indicator2=d['indicator2'].strip('.'), effect_type=self.effect_type_str().replace('  ',' ')) for  d in pos_examples_sample ]
                        neg_examples_open_ended_answer = [ template_responses[ens_idx][map_category_label['No']].format(indicator1=d['indicator1'].strip('.'), indicator2=d['indicator2'].strip('.'), effect_type=self.effect_type_str()).replace('  ',' ') for d in neg_examples_sample ]

                    # python shuffle two lists in the same order 
                    li_examples = list(zip( list(pos_examples_sample) + list(neg_examples_sample), list(pos_examples_open_ended_answer) + list(neg_examples_open_ended_answer) ))
                    random.shuffle(li_examples)

                    examples_sample, examples_open_ended_answer = zip(*li_examples)

                    # Creating the format dict for all examples
                    if self.relationship == 'budgetitem_to_indicator':
                        format_dict =  reduce(operator.ior, ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} ) # type: ignore
                    elif self.relationship == 'indicator_to_indicator':
                        format_dict =  reduce(operator.ior, ( { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} )
                else:
                    format_dict = {}

                # filling in the target info
                if self.relationship == 'budgetitem_to_indicator':
                    # prompt =  templates[ens_idx].format(target_budget_item= row['budget_item'], target_indicator=row['indicator'].strip('.'), effect_type=self.effect_type_str(),
                    #                                 **format_dict).replace('  ',' ')

                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_budget_item= row['budget_item'], target_indicator=row['indicator'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message"),
                        **format_dict
                         }
                elif self.relationship == 'indicator_to_indicator':
                    # prompt =  templates[ens_idx].format(target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.'), effect_type=self.effect_type_str(),
                    #                                 **format_dict).replace('  ',' ')
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message"),
                        **format_dict
                         }
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        # Generate the Open Response
        li_li_prompts_fmtd, li_li_statement = self.generate(li_li_prompts)
        li_li_discourse = [ [ prompt_fmtd+statement for prompt_fmtd, statement in zip(li_prompts, li_statement) ] for li_prompts, li_statement in zip(li_li_prompts_fmtd, li_li_statement) ]
        
        # Put the response in a categorical question template
        li_li_filled_template = self._fill_template_open_add_categorical_question_template(li_li_statement, batch)

        # Get Categorical Question Response
        li_li_prompts_fmtd, li_li_preds = self.generate(li_li_filled_template, include_user_message_pre_prompt=False, include_system_message=False, max_new_tokens=60, max_tokens=60 )

        # Extendinng the discourse with the categorical question and response

        li_li_discourse = [ [ ('===first section===\n\n' + discourse + '\n\n===second section===\n\n' + prompt_fmtd + pred) for discourse, prompt_fmtd, pred in zip(li_discourse, li_prompts, li_preds) ] for li_discourse, li_prompts, li_preds in zip(li_li_discourse, li_li_prompts_fmtd, li_li_preds) ]

        return li_li_preds, li_li_discourse
    
    def _fill_template_open_add_categorical_question_template(self, li_li_statement, batch):
        li_li_filled_templates = []

        for li_statements, row in zip(li_li_statement, batch):
            # Template to prompt language llm to simplify the answer to a Yes/No output
            li_template = map_relationship_promptsmap[self.relationship]['li_prompts_categorical_question_w_reasoning']
            template : dict = copy.deepcopy( random.choice(li_template) )     

            li_filledtemplate = []
            # Formatting the template
            for statement in li_statements:
                user_message = template['user_message'].format(statement=statement, effect_type=self.effect_type_str(), budget_item=row ['budget_item'], indicator=row['indicator'] )
                assistant_message = template['assistant_message']
                
                # filled_template = map_llmname_input_format(self.llm_name, user_message, response=assistant_message)
                filled_template = {
                    "user_message": user_message,
                    "assistant_message": assistant_message
                }

                li_filledtemplate.append(filled_template)

            
            li_li_filled_templates.append(li_filledtemplate)

        return li_li_filled_templates
        
    def fill_template_categorise(self, templates:list[dict], batch:list[dict], reverse_categories_order:bool=False)->tuple[list[list[str]], list[list[str]]]:

        """Fill in the template with the target and k_shot context"""

        li_li_prompts = []
        map_category_label = map_relationship_promptsmap[self.relationship]['map_category_label']
        
        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            categorise_response_labels_inverted = {v:k for k,v in map_category_label.items()}
       
            if reverse_categories_order is True:
                _categorise_response_labels_inverted = copy.deepcopy(categorise_response_labels_inverted)
                keys = list(_categorise_response_labels_inverted.keys())
                _ = _categorise_response_labels_inverted[keys[0]]
                _categorise_response_labels_inverted[keys[0]] = _categorise_response_labels_inverted[keys[1]] 
                _categorise_response_labels_inverted[keys[1]] = _                

            for ens_idx in range(self.ensemble_size):
                
                # This indented code section fills in the k_shot context with random extracts from dataset
                if self.k_shot == 0:
                    format_dict = {}
                # sample k items from our train set into a format dict for the template
                elif self.relationship == 'budgetitem_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":_categorise_response_labels_inverted[d['related']] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} ) 
                elif self.relationship == 'indicator_to_indicator':
                    raise NotImplementedError("Categorise response not implemented for indicator_to_indicator")
                    format_dict = reduce( operator.ior, [ { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}":['related'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} )
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    user_message = templates[ens_idx]["user_message"].format(
                        target_budget_item= row['budget_item'].strip('.'), target_indicator=row['indicator'].strip('.'),
                        **format_dict
                    )
                elif self.relationship == 'indicator_to_indicator':
                    user_message = templates[ens_idx]["user_message"].format(
                        target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.'),
                        **format_dict
                    )

                filled_template = {
                    "user_message": user_message,
                    "assistant_message": templates[ens_idx].get("assistant_message")
                }
                li_prompts.append(filled_template)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        li_filled_prompt = li_li_prompts
        return li_filled_prompt, []

    def fill_template_cot(self, templates:list[dict], batch:list[dict], reverse_categories_order:bool=False)->tuple[list[list[str]], list[list[str]]]:

        li_li_prompts = []

        # for each row in batch create the template for the first step of the COT
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            
            for ens_idx in range(self.ensemble_size):
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    # prompt = 
                    # )
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_budget_item= row['budget_item'].strip('.'), target_indicator=row['indicator'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message")
                         }


                elif self.relationship == 'indicator_to_indicator':
                    prompt = {
                        "user_message": templates[ens_idx]["user_message"].format(
                        target_indicator1= row['indicator1'].strip('.'), target_indicator2=row['indicator2'].strip('.')),     
                        "system_message": templates[ens_idx].get("system_message"),
                        "assistant_message": templates[ens_idx].get("assistant_message")
                         }
                    
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)

        # Now all prompts created get the 1st Step Chain of Thought response from the LLM
        li_li_prompts_fmtd, li_li_reasoning = self.generate(li_li_prompts)
        li_li_discourse = [ [ prompt_fmtd+reasoning for prompt_fmtd, reasoning in zip(li_prompts, li_reasoning) ] for li_prompts, li_reasoning in zip(li_li_prompts_fmtd, li_li_reasoning) ]

        # Now insert each llm response reason into the final template including the COT conversation
        li_li_filledtemplate = self._fill_template_cot_classifyconv( li_li_reasoning, batch, reverse_categories_order )
        
        return li_li_filledtemplate, li_li_discourse

    def _fill_template_cot_classifyconv(self, li_li_reasoning:list[list[str]], batch:list[dict], reverse_categories_order:bool=False, seed=None  )->list[list[str]]:
        if seed is not None:
            random.seed(seed)
        
        if not reverse_categories_order:
            li_templates = map_relationship_promptsmap[self.relationship]['li_prompts_categorical_question_w_reasoning']
        elif reverse_categories_order:
            li_templates = map_relationship_promptsmap[self.relationship]['li_prompts_categorical_question_w_reasoning_reversed']

        template: dict  = copy.deepcopy( random.choice(li_templates) )
        
        li_li_filledtemplate = []
        for li_reasoning, dict_datum in zip( li_li_reasoning, batch):
            # Filling template
            li_filledtemplate = []
            for pred in li_reasoning:
                if self.relationship == 'budgetitem_to_indicator':
                    user_message = template['user_message'].format(statement=pred, budget_item=dict_datum['budget_item'], indicator=dict_datum['indicator'].strip('.'), effect_type=self.effect_type_str() )
                    response = template['assistant_message']

                elif self.relationship == 'indicator_to_indicator':
                    user_message = template['user_message'].format(statement=pred, indicator1=dict_datum['indicator1'].strip('.'), indicator2=dict_datum['indicator2'].strip('.'), effect_type=self.effect_type_str() )
                    response = template['assistant_message']

                # filled_template = map_llmname_input_format(self.llm_name, user_message, response=response)

                filled_template = {
                    "user_message": user_message,
                    "assistant_message": response
                }

                li_filledtemplate.append(filled_template)


            li_li_filledtemplate.append(li_filledtemplate)
        
        return li_li_filledtemplate

    def _fill_template_categories_scale(self, templates, batch ) -> tuple[list[list[str]], list[list[str]]]:
        
        """ fill in the template with the indicators """

        li_li_prompts = []
        
        for row in batch:

            li_prompts = []
            prompt = None

            for ens_idx in range(len(templates)):

                if self.relationship == 'budgetitem_to_indicator':
                    raise NotImplementedError("Categories scale question not implemented for budgetitem_to_indicator relationship")

                elif self.relationship == 'indicator_to_indicator':
                    # prompt = templates[ens_idx].format(
                    #     target_indicator1= row['indicator1'].strip('.'), 
                    #     target_indicator2=row['indicator2'].strip('.')
                    # )
                    # li_prompts.append(prompt)
                    filled_template = {
                        "user_message": templates[ens_idx]['user_message'].format(
                            target_indicator1= row['indicator1'].strip('.'), 
                            target_indicator2=row['indicator2'].strip('.')
                        ),
                        "assistant_message": None
                    }
                li_prompts.append(filled_template)

            li_li_prompts.append(li_prompts)
        
        li_filled_prompt = li_li_prompts
        li_discourse = [] # No discourse was needed for this output

        return li_filled_prompt, li_discourse

    def _fill_template_verbalize_scale(self, templates, batch, **kwargs ) -> tuple[list[list[str]], list[list[str]]]:
        
        """ fill in the template with the indicators """

        li_li_prompts = []
        
        for row in batch:

            li_prompts = []
            prompt = None

            for ens_idx in range(self.ensemble_size):

                if self.relationship == 'budgetitem_to_indicator':
                    raise NotImplementedError("Scale question not implemented for budgetitem_to_indicator relationship")

                elif self.relationship == 'indicator_to_indicator':
                    prompt = {
                        "user_message": templates[ens_idx]['user_message'].format(
                        target_indicator1= row['indicator1'].strip('.'), 
                        target_indicator2=row['indicator2'].strip('.')
                    )
                    } 
                    
                li_prompts.append(prompt)
                
            li_li_prompts.append(li_prompts)
        

        li_li_prompts_fmtd, li_li_preds = self.generate(li_li_prompts, add_suffix_space=True, **kwargs)

        li_li_statement = li_li_preds
        li_li_discourse = [ [ prompts_fmtd+pred for prompts_fmtd, pred in zip(li_prompts, li_preds) ] for li_prompts, li_preds in zip(li_li_prompts_fmtd, li_li_preds) ]
        
        return li_li_statement, li_li_discourse