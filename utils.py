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

logger = logging.getLogger(__name__)

from env import apikey_ark, apikey_tencent, apikey_ds

ark_models_entrypoints= {
    'ds-r1-7b':"ep-20250218161643-vl25s",
    'ds-r1-7b_Batch':"ep-bi-20250401151011-nzp5d",
    'ds-r1-32b':"ep-20250218161612-h68p4",
    'ds-r1-32b_Batch':"ep-bi-20250401145433-t26nc",
    'ds-v3': "ep-20250214165420-s2s6s",
    'ds-v3_Batch':"ep-bi-20250401151053-tzcz4",
    'ds-r1': "ep-20250214165338-52bjm",
    'ds-r1_Batch':"ep-bi-20250401151117-zwvf8",
    'doubao-lite':"ep-20241217204945-k7pmp",
    'doubao-pro':"ep-20241012144130-85pt2",
    'doubao-pro_Batch':"ep-bi-20250402103826-ll9wf",
}


url_ark = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
url_ark_batch = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
url_ds = "https://api.deepseek.com/chat/completions"
url_tencent = 'https://api.lkeap.cloud.tencent.com/v1/chat/completions'
url_ollama = "http://localhost:11434/api/chat"

MODEL_MAPPING_TENCENT = {
    'ds-r1':"deepseek-r1",
    'ds-v3-old':"deepseek-v3",
    'ds-v3':"deepseek-v3-0324",
}
MODEL_MAPPING_DEEPSEEK = {
    'ds-r1':"deepseek-reasoner",
    'ds-v3':"deepseek-chat",
}

MODEL_MAPPING_OLLAMA = {
    'qwen2.5':"qwen2.5:latest",
    'qwen2.5-32b':"qwen2.5:32b",
    'deepseek-r1':"deepseek-r1:latest",
    'deepseek-r1-32b':"deepseek-r1:32b",
}




def chat_llm(questions,model_config,max_try=3):
    result = []
    if model_config['server'] == 'ollama':
        for question in questions:
            response = ollama_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
            if response is not None:
                result.append(response)
            else:
                # retry 3 times
                for _ in range(max_try):
                    print(f"retry {_} for question: {question}")
                    response = ollama_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
                    if response is not None:
                        result.append(response)
                        break
                    time.sleep(10)
    elif model_config['server'] == 'deepseek':
        for question in questions:
            response = deepseek_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
            if response is not None:
                result.append(response)
            else:
                # retry 3 times
                for _ in range(max_try):
                    print(f"retry {_} for question: {question}")
                    response = deepseek_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
                    if response is not None:
                        result.append(response)
                        break
    elif model_config['server'] == 'tencent':
        for question in questions:
            response = tencent_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
            if response is not None:
                result.append(response)
            else:
                # retry 3 times
                for _ in range(max_try):
                    print(f"retry {_} for question: {question}")
                    response = tencent_chat(question,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
                    if response is not None:
                        result.append(response)
                        break
                    time.sleep(10)
    return result

async def async_chat_llm(questions,model_config):
    result = []
    
    if model_config['server'] == 'deepseek':
        results = await batch_deepseek_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    elif model_config['server'] == 'tencent':
        results = await batch_tencent_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    elif model_config['server'] == 'ark':
        results = await batch_ark_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    elif model_config['server'] == 'ollama':
        results = await batch_ollama_chat_async(questions,model_config['temperature'],model_config['model'],timeout=model_config['timeout'])
    return results





def ollama_chat(question="你好,你是谁？",temperature=0.5,model="qwen2.5",url=url_ollama,timeout=1800):
        
    headers = {
        'Content-Type': 'application/json',
    }
        
    data = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature":temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response is None:
        return None
    if response.status_code != 200:
        return None
    response = response.json()
    result = {}
    result_raw = response['message'].get('content','')
                # 提取 字符中的<thinking>...</thinking> 
    reasoning = re.findall(r'<thinking>(.*?)</thinking>', result_raw, re.DOTALL)
    reasoning = reasoning[0] if reasoning else ''
    token_count = response['eval_count'] + response['prompt_eval_count']
    result['reasoning_content'] = reasoning
    result['total_tokens']= token_count
    result['content'] = result_raw
    return result

async def ollama_chat_async(question="你好,你是谁？",temperature=0.5,model="qwen2.5",url=url_ollama,timeout=1800):
    # TODO

    return None

async def batch_ollama_chat_async(questions: List[str], temperature=0.5, model="qwen2.5", timeout=1800) -> List[Tuple[str, str]]:
    # TODO
    return [None] * len(questions)


def ark_chat(question="你好,你是谁？",temperature=0.5,model="ds-r1-7b_Batch",url=url_ark_batch,apikey=apikey_ark):
        
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
        
    data = {
        "model": ark_models_entrypoints[model],
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": " "
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature":temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response is None:
        return None
    if response.status_code != 200:
        return None
    return response.json()

    
def tencent_chat(question="你好,你是谁？",temperature=0,model="ds-r1",url=url_tencent,apikey=apikey_tencent):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    data = {
        "model": MODEL_MAPPING_TENCENT[model],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response is None:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def deepseek_chat(question="你好,你是谁？",temperature=0,model="ds-r1",url=url_ds,apikey=apikey_ds,timeout=3000):
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    data = {
        "model": MODEL_MAPPING_DEEPSEEK[model],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data),timeout=timeout)
    if response is None:
        return None
    if response.status_code != 200:
        return None
    return response.json()


async def tencent_chat_async(
    question: str = "你好,你是谁？", 
    temperature: float = 0, 
    model: str = "ds-r1", 
    url: str = url_tencent, 
    apikey: str = apikey_tencent, 
    timeout: int = 1800
):
    """
    异步调用腾讯聊天模型API
    
    Args:
        question: 用户问题
        temperature: 温度参数，控制回答的随机性
        model: 模型代号，支持ds-r1, ds-v3_old, ds-v3
        url: API请求地址
        apikey: API密钥
        timeout: 请求超时时间(秒)
        
    Returns:
        API响应JSON或None(出错时)
    """
    # 验证模型
    if model not in MODEL_MAPPING_TENCENT:
        print(f"不支持的模型: {model}")
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    
    data = {
        "model": MODEL_MAPPING_TENCENT[model],
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
    
    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            # 使用重试装饰器包装请求操作
            # @with_retry(max_retries=3)
            async def make_request():
                async with session.post(url, headers=headers, json=data) as response:
                    return response
                    
            response = await make_request()
            
            if response is None or response.status != 200:
                error_msg = f"HTTP错误: {response.status if response else 'Unknown'}"
                print(error_msg)
                return None
                
            return await response.json()
                
    except asyncio.TimeoutError:
        print(f"请求超时: {question[:50]}...")
        return None
    except Exception as e:
        print(f"处理请求出错: {question[:50]}...")
        return None


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
    if model not in MODEL_MAPPING_DEEPSEEK:
        print(f"不支持的模型: {model}")
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    
    data = {
        "model": MODEL_MAPPING_DEEPSEEK[model],
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
    if model not in ark_models_entrypoints:
        raise ValueError(f"Model {model} not found in ark_models_entrypoints")
    data = {
        "model": ark_models_entrypoints[model],
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


async def batch_tencent_chat_async(questions: List[str], temperature=0, model="ds-r1", timeout=1200) -> List[Tuple[str, str]]:
    """
    Process multiple questions asynchronously
    Returns a list of tuples containing (response, reasoning) for each question
    """
    tasks = []
    for question in questions:
        task = tencent_chat_async(question=question, temperature=temperature, model=model, timeout=timeout)
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
