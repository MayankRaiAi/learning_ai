"""
Application configuration settings.
"""
import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Ensure the current directory is also in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from secret_keys import OPENAI_API_KEY

from langchain_openai import ChatOpenAI


# OpenAI model configuration
MODEL_NAME = "gpt-3.5-turbo"
HIGH_TEMPERATURE = 0.9
LOW_TEMPERATURE = 0.2

def jolly_llm(model: str=MODEL_NAME, temperature: int=HIGH_TEMPERATURE):
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature
    )