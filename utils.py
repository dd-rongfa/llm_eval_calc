import requests
import json
import asyncio
import aiohttp
from typing import List, Tuple
import ollama
import logging
import re
import time
from typing import Optional, Dict, Any, Callable, TypeVar
from functools import wraps
import random
import openai 
from openai import OpenAI,AsyncOpenAI

from env import apikey_ark, apikey_ds



# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


MODEL_CONFIG = {
    'ark': {
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3/',
        'api_key': apikey_ark,
        'models': {
            'ds-r1-7b': "ep-20250218161643-vl25s",
            'ds-r1-32b': "ep-20250218161612-h68p4",
            'ds-v3': "ep-20250214165420-s2s6s",
            'ds-r1': "ep-20250214165338-52bjm",
            'doubao-lite': "ep-20241217204945-k7pmp",
            'doubao-pro': "ep-20241012144130-85pt2"
        }
    },
    'ark_batch': {
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3/batch/',
        'api_key': apikey_ark,
        'models': {
            'ds-r1-7b': "ep-bi-20250401151011-nzp5d",
            'ds-r1-32b': "ep-bi-20250401145433-t26nc",
            'ds-v3': "ep-bi-20250401151053-tzcz4",
            'ds-r1': "ep-bi-20250401151117-zwvf8",
            'doubao-pro': "ep-bi-20250402103826-ll9wf"
        }
    },
    'deepseek': {
        'base_url': "https://api.deepseek.com",
        'api_key': apikey_ds,
        'models': {
            'ds-r1': "deepseek-reasoner",
            'ds-v3': "deepseek-chat"
        }
    },
    'ollama': {
        'base_url': "http://localhost:11434/api/chat",
        'api_key': "ollama",
        'models': {
            'qwen2.5': "qwen2.5:latest",
            'qwen2.5-32b': "qwen2.5:32b",
            'deepseek-r1': "deepseek-r1:latest",
            'deepseek-r1-32b': "deepseek-r1:32b"
        }
    }
}


def openai_chat(question,client,model_config):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': question}],
            model=model_config['model'],
            temperature=model_config['temperature'],
            max_tokens=model_config['max_tokens'],
            timeout=model_config['timeout']
        )
        if chat_completion is None:
            logger.warning(f"llm chat error got None: {question} ")
            return None
        result = {}
        answer = chat_completion.choices[0].message.content
        reasoning = getattr(chat_completion.choices[0].message, 'reasoning_content', '')
        token_count = chat_completion.usage.total_tokens
        result['reasoning_content'] = reasoning
        result['total_tokens'] = token_count
        result['content'] = answer
        return result

    except Exception as e:
        logger.warning(f"Error in openai_chat: {question}  :{str(e)}")
        return None
    
   

async def async_openai_chat(question,client,model_config):
    try:
        chat_completion = await client.chat.completions.create(
                model=model_config['model'],
                messages=[{"role": "user", "content": question}],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens'],
                timeout=model_config['timeout']
            )
        if chat_completion is None:
            return None
            
        result = {}
        result['content'] = chat_completion.choices[0].message.content
        result['reasoning_content'] = getattr(chat_completion.choices[0].message, 'reasoning_content', '')
        result['total_tokens'] = chat_completion.usage.total_tokens
        return result
    except Exception as e:
        logger.warning(f"Error in async_chat_llm: {str(e)}")
        return None



def chat_llm_batch(questions, model_config):
    result = []
    
    server = model_config['server']
    if server not in MODEL_CONFIG:
        logger.warning(f"不支持的服务器类型: {server}")
        raise ValueError(f"不支持的服务器类型: {server}")
    model_config['base_url'] = MODEL_CONFIG[server]['base_url']
    model_config['model'] = MODEL_CONFIG[server]['models'][model_config['model_name']]
    model_config['api_key'] = MODEL_CONFIG[server]['api_key']
    client = OpenAI(
        base_url=model_config['base_url'],
        # required but ignored
        api_key=model_config['api_key'],
    )
   
    for question in questions:
        # 首次尝试
        response = openai_chat(question,client,model_config)
        # 如果成功，添加结果并继续
        if response is not None:
            result.append(response)
            continue
    return result


async def async_chat_llm_batch(questions, model_config):
    
    server = model_config['server']
    if server not in MODEL_CONFIG:
        logger.warning(f"不支持的服务器类型: {server}")
        raise ValueError(f"不支持的服务器类型: {server}")
    model_name = model_config['model_name']
    if model_name not in MODEL_CONFIG[server]['models']:
        print(f"不支持的模型: {model_name}")
        raise ValueError(f"不支持的模型: {model_name}")
    model_config['base_url'] = MODEL_CONFIG[server]['base_url']
    model_config['model'] = MODEL_CONFIG[server]['models'][model_name]
    model_config['api_key'] = MODEL_CONFIG[server]['api_key']
    

    client = AsyncOpenAI(
        base_url=model_config['base_url'],
        # required but ignored
        api_key=model_config['api_key'],
    )

    tasks = []
    for question in questions:
        task = async_openai_chat(question=question, client=client, model_config=model_config)
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results



async def main_async(config):
    questions = [
        "你是谁？",
        "4*8=?"
    ]
    # results = await batch_ark_chat_async(questions)
    results = await async_chat_llm_batch(questions,model_config=config)
    print(results)

def main(config):
    questions = [
        "你是谁？",
        "4*8=?"
    ]
    results = chat_llm_batch(questions,model_config=config)
    print(results)

    
if __name__ == "__main__":
   
   # test
   config = {
    'server': 'deepseek',
    'model_name': 'ds-r1',
    'temperature': 0,
    'timeout': 1200,
    'max_tokens': 8192
   }
   ## test ok
   # asyncio.run(main_async(config))
   main(config)
