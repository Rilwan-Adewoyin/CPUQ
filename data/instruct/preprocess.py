import os
from transformers import BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from argparse import ArgumentParser

from utils import HUGGINGFACE_MODELS
from prompt_engineering.utils_prompteng import map_llmname_input_format
from prompt_engineering.my_logger import setup_logging_preprocess


def main(model_id, json_file, max_tokens_per_chunk=None):

    # Setup logging
    logging = setup_logging_preprocess( json_file, model_id )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data from JSON file
    raw_dataset = load_dataset('json', data_files=os.path.join('data','instruct', json_file) )

    # Shuffle and split data
    raw_dataset = raw_dataset.shuffle()
    dataset_dict = raw_dataset['train'].train_test_split(train_size=0.8)

    # Process and tokenize data
    for key in dataset_dict:
        # Apply custom function to each data instance
        dataset_dict[key] = dataset_dict[key].map(lambda batch: format_for_lm(batch, model_id, json_file), batched=False)

        # Tokenize data and create labels
        dataset_dict[key] = dataset_dict[key].map(lambda batch: tokenize_create_labels(batch, tokenizer, max_len=max_tokens_per_chunk), batched=False)

        # # Filter out instances where all elements of 'labels' are -100
        # dataset_dict[key] = dataset_dict[key].filter(lambda example: not all(label == -100 for label in example['labels']))

    # Save data to arrow files
    dir_ = './data/instruct/preprocessed'
    os.makedirs(dir_, exist_ok=True)
    
    fn = json_file.split('.')[0]

    dataset_dict['train'].set_format(type='torch', columns=["input_ids", "attention_mask", "labels"] )
    dataset_dict['test'].set_format(type='torch', columns=[ "input_ids", "attention_mask", "labels"] ) 

    # dataset_dict['train'].split = 'train'
    # dataset_dict['test'].split = 'test'

    # dataset_dict['train'].dataset_size = len(dataset_dict['train'])
    # dataset_dict['test'].dataset_size = len(dataset_dict['test'])

    
    dataset_dict['train'].save_to_disk(os.path.join(dir_, f'{fn}_{model_id.replace("/","_")}_train.arrow'))
    dataset_dict['test'].save_to_disk(os.path.join(dir_, f'{fn}_{model_id.replace("/","_")}_test.arrow'))

    logging.info('Finished Preprocessing Data')

def format_for_lm(data, llm_name, json_file):
    """Apply the function 'map_llmname_input_format' to each data instance.
        This formats the data in the way that the LLM was trained on. (for chat/instruct LLMs)
    """
    # Modify the following line according to your function
    if json_file == 'wLM70k_nofilt.json':
        system_message = None
        user_message = data['instruction']
        response = data['output']

    
    text = map_llmname_input_format(llm_name, system_message=system_message, user_message=user_message, response=response).strip(' ')

    return {'text':text}

def tokenize_create_labels(batch, tokenizer, max_len:int|None=None):
    # Tokenize each row of the dataset
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    outp = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len, return_offsets_mapping=True )
    
    # Create labels for masked language modeling
    labels = [-100] * len(outp['input_ids']) # Initialize labels with -100

    # Find the start and end character position of the 'output' in the original text

    output_start_char_pos = batch['text'].index(batch['output'])

    
    output_end_char_pos = output_start_char_pos + len(batch['output'])

    # Find the token start index and token end index that map to the start and end character position of the 'output'
    try:
        output_start_token_index = next(i for i, (start_pos, end_pos) in enumerate(outp['offset_mapping']) if start_pos <= output_start_char_pos < end_pos)
    except StopIteration:
        output_start_token_index = len(outp['input_ids'])

    try:
        output_end_token_index = next(i for i, (start_pos, end_pos) in enumerate(outp['offset_mapping']) if start_pos < output_end_char_pos <= end_pos)
    except StopIteration:
        output_end_token_index = len(outp['input_ids'])

    # Replace the labels from token start index to token end index with the corresponding 'input_ids'
    # We do + 1 to account for the eos_token that needs to be predicted
    labels[output_start_token_index:(output_end_token_index+1)+1] = outp['input_ids'][output_start_token_index:(output_end_token_index+1)+1]

    # labels = labels[1:] + [-100]  # shift labels to the left, append -100 to the end

    outp['labels'] = labels

    outp.pop('offset_mapping')
    # outp.update(batch)

    return outp

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--model_id', type=str, default='TheBloke/Wizard-Vicuna-13B-Uncensored-HF', choices = HUGGINGFACE_MODELS )
    
    parser.add_argument('--json_file', type=str, default='wLM70k_nofilt.json', choices=['wLM70k_nofilt.json'])

    parser.add_argument('--max_tokens_per_chunk', type=int, default=256)

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()


    main(**vars(args))

