import os
import pytest

from langchain import HuggingFacePipeline
from  langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from utils import  HUGGINGFACE_MODELS, OPENAI_MODELS, PredictionGenerator, ALL_MODELS, MAP_LOAD_IN_8BIT
from prompt_engineering.predict import load_llm
import yaml

# Set environment variables

with open('test/env_vars.yaml', 'r') as file:
    for k, v in yaml.safe_load(file).items():
        os.environ[k] = v

@pytest.mark.parametrize("llm_name, finetuned, local_or_remote, api_key, prompt_style", [
    # ("mosaicml/mpt-7b-chat", True, "local", None, "yes_no"),
    ("mosaicml/mpt-7b-chat", False, "local", None, "yes_no"),
    ("ausboss/llama-30b-supercot", False, "local", None, "yes_no"),
    ("TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g", False, "local", None, "yes_no"),

    # ("mosaicml/mpt-7b-chat", True, "local", None, "yes_no"),
    # ("mosaicml/mpt-7b-chat", False, "remote", "my-api-key", "yes_no"),
    # ("mosaicml/mpt-7b-chat", True, "remote", "my-api-key", "yes_no"),
])

def test_load_llm(llm_name, finetuned, local_or_remote, api_key, prompt_style):
    """Tests the load_llm() function."""

    # Assert that the arguments are valid.
    assert local_or_remote in ["local", "remote"], f"local_or_remote must be either 'local' or 'remote', not {local_or_remote}"
    if local_or_remote == "remote":
        assert api_key is not None, f"api_key must be provided if local_or_remote is 'remote'"
    if local_or_remote == "local":
        assert llm_name not in OPENAI_MODELS, f"llm_name must be a HuggingFace model if local_or_remote is 'local'"
    assert llm_name in ALL_MODELS, f"llm_name must be a valid model name, not {llm_name}"

    # Call the load_llm() function.
    llm, tokenizer = load_llm(llm_name, finetuned, local_or_remote, api_key, prompt_style)

    # Assert that the llm object is not None.
    assert llm is not None

    # Assert that the llm object has the correct attributes.
    if local_or_remote == "local":
        assert llm.model_id == llm_name
        assert llm.task == "text-generation"
        assert llm.model_kwargs["max_new_tokens"] == 5 if prompt_style == "yes_no" else 50
        assert llm.model_kwargs["load_in_8bit"] == MAP_LOAD_IN_8BIT[llm_name]
        assert llm.model_kwargs["device_map"] == "auto"
    else:
        assert llm.repo_id == llm_name
        assert llm.huggingfacehub_api_token == api_key
        assert llm.model_kwargs["max_new_tokens"] == 5 if prompt_style == "yes_no" else 100
        assert llm.model_kwargs["do_sample"] == False

