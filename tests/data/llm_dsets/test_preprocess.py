import unittest
from transformers import AutoTokenizer
from datasets import Dataset

from prompt_engineering.data.llm_dsets.preprocess import format_for_lm, map_tokenize, create_labels_with_mask

class TestPreprocessingMethods(unittest.TestCase):

    def setUp(self):
        self.data = {'instruction': 'Hello world!', 'output': 'Goodbye world!'}
        self.llm_name = 'julien-c/dummy-unknown'
        self.json_file = 'wLM70k_nofilt.json'
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        self.dataset = Dataset.from_dict(self.data)

    def test_format_for_lm(self):
        result = format_for_lm(self.data, self.llm_name, self.json_file)
        self.assertIn('text', result)

    def test_map_tokenize(self):
        batch = {'text': ['Hello world!']}
        result = map_tokenize(batch, self.tokenizer)
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)

    def test_create_labels_with_mask(self):
        batch = {
            'output': 'Hello world!',
            'input_ids': self.tokenizer.encode('Hello world!'),
            'attention_mask': self.tokenizer.encode('Hello world!')
        }
        result = create_labels_with_mask(batch, self.tokenizer)
        self.assertIn('labels', result)

if __name__ == '__main__':
    unittest.main()
