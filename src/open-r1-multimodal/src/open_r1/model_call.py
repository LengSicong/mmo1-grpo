import base64
import json
import os
import ast
import random
import shutil
import numpy as np
from multiprocessing import Pool, Manager
import openai
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import time
import logging
from PIL import Image
import io

# Set up a basic configuration for logging
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("model_call.py")

logger.setLevel(logging.DEBUG)

# Create the client objects
client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")
embedding_client = OpenAI(base_url="http://localhost:30001/v1", api_key="None")
llm_client = OpenAI(base_url="http://localhost:30002/v1", api_key="None")

# Function to handle OpenAI Chat Completions with error handling
def VisionChatCompletions(base64imgs, query_system, query, num_samples=1):
    retries = 5
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model="mmo1-72b", 
                temperature=0.5,
                top_p=0.9,
                # stop="red",
                n=num_samples,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [                        
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}}, base64imgs),
                        {"type": "text", "text": query},
                    ]}
                ]
            )
            return completion
        except openai.OpenAIError as e:
            # print(f'ERROR: {e}')
            logger.error(f'{e} in VisionChatCompletions')
            return f"Unsuccessful: {e.message}"
    
    # print(f"Failed after multiple retries.")
    logger.error(f"Failed after multiple retries in VisionChatCompletions")
    return f"Unsuccessful: Failed after multiple retries."

# Function to handle OpenAI Chat Completions with error handling
def VisionChatCompletions_greedy(base64imgs, query_system, query, num_samples=1):
    retries = 5
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model="mmo1-72b", 
                temperature=0.,
                top_p=0.15,
                # stop="red",
                n=num_samples,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [                        
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}}, base64imgs),
                        {"type": "text", "text": query},
                    ]}
                ]
            )
            return completion
        except openai.OpenAIError as e:
            # print(f'ERROR: {e}')
            logger.error(f'{e} in VisionChatCompletions')
            return f"Unsuccessful: {e.message}"
    
    # print(f"Failed after multiple retries.")
    logger.error(f"Failed after multiple retries in VisionChatCompletions")
    return f"Unsuccessful: Failed after multiple retries."


# Function to handle OpenAI Chat Completions with error handling
def VisionChatCompletions_single_step(base64imgs, query_system, query, num_samples=1):
    retries = 5
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model="mmo1-72b", 
                temperature=0.5,
                top_p=0.9,
                # stop="red",
                max_tokens=1024,
                n=num_samples,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [                        
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}}, base64imgs),
                        {"type": "text", "text": query},
                    ]}
                ]
            )
            return completion
        except openai.OpenAIError as e:
            logger.error(f'{e} in VisionChatCompletions_single_step')
            return f"Unsuccessful: {e.message}"
    
    logger.error(f"Failed after multiple retries in VisionChatCompletions_single_step")
    return f"Unsuccessful: Failed after multiple retries."

def TextChatCompletions(query_system, query):
    """
    Function to handle OpenAI Chat Completions with error handling
    """
    retries = 5
    for _ in range(retries):
        try:
            completion = llm_client.chat.completions.create(
                model="Qwen2.5-7B", 
                temperature=0,
                top_p=0.1,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": query}
                ]
            )
            return completion
        except openai.OpenAIError as e:
            logger.error(f'{e} in TextChatCompletions')
            return f"Unsuccessful: {e.message}"
    
    logger.error(f"Failed after multiple retries in TextChatCompletions")
    return f"Unsuccessful: Failed after multiple retries."

def TextEmbeddingCompute(queries: list):
    """
    Function to get text embeddings for a list of queries.
    """
    retries = 5
    for _ in range(retries):
        try:
            embeddings = embedding_client.embeddings.create(
                input=queries,
                model='Qwen2-1.5B-embedding',
            )
            return embeddings
        except openai.OpenAIError as e:
            logger.error(f'{e} in TextEmbeddingCompute')
            return f"Unsuccessful: {e.message}"
    
    logger.error(f"Failed after multiple retries in TextEmbeddingCompute")
    return f"Unsuccessful: Failed after multiple retries."

# def img2base64(img_path):
#     with open(img_path, "rb") as img_file:
#         img_str = base64.b64encode(img_file.read()).decode('utf-8')
#     return img_str

def img2base64(img_path):
    with Image.open(img_path) as img:
        img_size = img.size
        if img_size[0] < 28 or img_size[1] < 28:
            ratio = max(28 / img_size[0], 28 / img_size[1])
            new_size = (int(np.ceil(img_size[0] * ratio)), int(np.ceil(img_size[1] * ratio)))
            img = img.resize(new_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG") 
        img_bytes = buf.getvalue()
        img_str = base64.b64encode(img_bytes).decode("utf-8")

    return img_str