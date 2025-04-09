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
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from env import apikey_ark, apikey_tencent, apikey_ds



MODEL_CONFIG = {
    'ark': {
        'url': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
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
        'url': 'https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions',
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
        'url': "https://api.deepseek.com/chat/completions",
        'api_key': apikey_ds,
        'models': {
            'ds-r1': "deepseek-reasoner",
            'ds-v3': "deepseek-chat"
        }
    },
    'ollama': {
        'url': "http://localhost:11434/api/chat",
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
        response = client.chat.completions.create(
            messages=[
                {
                'role': 'user',
                'content': question,
            }
        ],
        model=model_config['model'],
        temperature=model_config['temperature'],
            max_tokens=model_config['max_tokens'],
            timeout=model_config['timeout']
        )
    except Exception as e:
        logger.error(f"llm chat error: {question}")
        return None
    if response is None:
        logger.error(f"llm chat error: {question}")
        return None
    response = response.json()
    result = {}
    result_raw = response['choices'][0]['message'].get('content','')
                # 提取 字符中的<thinking>...</thinking> 
    reasoning = re.findall(r'<thinking>(.*?)</thinking>', result_raw, re.DOTALL)
    reasoning = reasoning[0] if reasoning else ''
    token_count = response['usage']['total_tokens']
    result['reasoning_content'] = reasoning
    result['total_tokens']= token_count
    result['content'] = result_raw
    return result

def chat_llm(questions, model_config, max_try=3):
    result = []
    
    server = model_config['server']
    if server not in MODEL_CONFIG:
        raise ValueError(f"不支持的服务器类型: {server}")
    model_config['url'] = MODEL_CONFIG[server]['url']
    model_config['model'] = MODEL_CONFIG[server]['models'][model_config['model_name']]

    client = OpenAI(
        base_url=model_config['url'],
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
            
        # 失败时使用指数退避重试
        for attempt in range(max_try):
            delay = min(10 * (2 ** attempt) + random.uniform(0, 1), 60)
            logger.info(f"retry {attempt+1}/{max_try} for question: {question}, waiting {delay:.2f} seconds")
            time.sleep(delay)
            response = openai_chat(question,client,model_config)
            if response is not None:
                result.append(response)
                break
        else:
            # 所有重试都失败时记录
            logger.info(f"所有重试都失败，问题: {question}")
            # 可选: result.append(None) 保持结果列表和问题列表长度一致
            result.append(None)
    
    return result


async def async_chat_llm(questions,model_config):
    result = []
    
    if model_config['server'] == 'deepseek':
        results = await batch_deepseek_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    elif model_config['server'] == 'ark':
        results = await batch_ark_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    elif model_config['server'] == 'ollama':
        results = await batch_ollama_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    return results




async def ollama_chat_async(question="你好,你是谁？",temperature=0.5,model="qwen2.5",url=url_ollama,timeout=1800):
    # TODO

    return None

async def batch_ollama_chat_async(questions: List[str], temperature=0.5, model="qwen2.5", timeout=1800) -> List[Tuple[str, str]]:
    # TODO
    return [None] * len(questions)



async def deepseek_chat_async(
    question: str = "你好,你是谁？", 
    temperature: float = 0.1, 
    model: str = "ds-r1", 
    url: str = url_ds, 
    apikey: str = apikey_ds, 
    timeout: int = 1800,
    max_retries: int = 3
):
    """
    异步调用Deepseek聊天模型API
    
    Args:
        question: 用户问题
        temperature: 温度参数，控制回答的随机性
        model: 模型代号，支持ds-r1, ds-v3
        url: API请求地址
        apikey: API密钥
        timeout: 请求超时时间(秒)
        max_retries: 最大重试次数
        
    Returns:
        API响应JSON或None(出错时)
    """
    # 验证模型
    if model not in MODEL_CONFIG['deepseek']['models']:
        print(f"不支持的模型: {model}")
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    
    data = {
        "model": MODEL_CONFIG['deepseek']['models'][model]['model_name'],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
    }
    
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = {}
                        response_json = await response.json()
                        result['content'] = response_json['choices'][0]['message'].get('content','')
                        result['reasoning_content'] = response_json['choices'][0]['message'].get('reasoning_content','')
                        result['total_tokens'] = response_json['usage']['total_tokens']
                        return result
                    elif response.status in (429, 500, 503):
                        wait_time = (attempt + 1) * 60  # 递增等待时间
                        print(f"HTTP {response.status}, 等待 {wait_time}秒后重试 ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"HTTP错误: {response.status}")
                        return None
                        
        except aiohttp.ClientConnectionError as e:
            print(f"连接错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 30)  # 递增等待时间
                continue
            return None
        except asyncio.TimeoutError:
            print(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 30)
                continue
            return None
        except Exception as e:
            print(f"处理请求出错: {str(e)}")
            return None
    return None

async def ark_chat_async(question="你好,你是谁？", temperature=0, model="ds-r1-7b_Batch",url=url_ark_batch,apikey=apikey_ark, timeout=1800):
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    # 检查model是否存在
    if model not in MODEL_CONFIG['ark']['models']:
        raise ValueError(f"Model {model} not found in ark_models_entrypoints")
    data = {
        "model": MODEL_CONFIG['ark']['models'][model]['endpoint'],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
    }
    
    timeout_obj = aiohttp.ClientTimeout(total=timeout)  # 设置总超时时间
    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(url, headers=headers, json=data) as response:
                # If code is 500,sleep 60s and retry
                if response.status in (500,503, 429):
                    print(f"Error: HTTP {response.status} sleep 120s and retry")
                    await asyncio.sleep(120) # 120s
                    return await ark_chat_async(question, temperature, model, timeout)
                if response.status != 200:
                    print(f"Error: HTTP {response.status}")
                    return None
                else:
                    result = {}
                    response_json = await response.json()

                    result['content'] = response_json['choices'][0]['message'].get('content','')
                    result['reasoning_content'] = response_json['choices'][0]['message'].get('reasoning_content','')
                    result['total_tokens'] = response_json['usage']['total_tokens']
                    return result
    except asyncio.TimeoutError:
        print(f"Timeout error for question: {question[:50]}...")
        return None
    except Exception as e:
        print(f"Error processing question: {question[:50]}... Error: {str(e)}")
        return None


async def batch_ark_chat_async(questions: List[str], temperature=0, model="ds-r1-7b_Batch", timeout=1800) -> List[Tuple[str, str]]:
    """
    Process multiple questions asynchronously
    Returns a list of tuples containing (response, reasoning) for each question
    """
    tasks = []
    for question in questions:
        task = ark_chat_async(question=question, temperature=temperature, model=model, timeout=timeout)
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Filter out any exceptions and convert them to empty results
    return [None if isinstance(r, Exception) else r for r in results]


async def batch_deepseek_chat_async(questions: List[str], temperature=0, model="ds-r1", timeout=1200) -> List[Tuple[str, str]]:
    """
    Process multiple questions asynchronously
    Returns a list of tuples containing (response, reasoning) for each question
    """
    tasks = []
    for question in questions:
        task = deepseek_chat_async(question=question, temperature=temperature, model=model, timeout=timeout)
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Filter out any exceptions and convert them to empty results
    return [None if isinstance(r, Exception) else r for r in results]

async def main_async():
    questions = [
        "你是谁？",
        "你能做什么？",
        "4*8=?"
    ]
    # results = await batch_ark_chat_async(questions)
    results = await batch_deepseek_chat_async(questions,model="ds-r1")
    print(results)



if __name__ == "__main__":
   
    # tencent chat test
    # res = tencent_chat("你是谁？",model="ds-v3",temperature=0)
    # print(res)

    # deepseek chat test
    # res = deepseek_chat("你是谁？",model="ds-v3",temperature=0)
    # print(res)

    # ark_chat
    res = ark_chat("你是谁？",model="doubao-pro_Batch",temperature=0)
    print(res)
