import requests
import json
import asyncio
import aiohttp
from typing import List, Tuple
import ollama

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
}

ark_url = "https://api.ark.com/v1/chat/completions"
apikey = "d0a38cca-7ef4-46ff-bda6-6433191af8eb"




def llm_chat(prompt, model="qwen2.5:latest",temperature=0):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           options={"temperature": temperature})
    return response["message"]


def ark_chat(question="你好,你是谁？",temperature=0.5,model="ds-r1-7b_Batch"):
        
    url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
    url_batch = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
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
    response = requests.post(url_batch, headers=headers, data=json.dumps(data)).json()
    
    result = response['choices'][0]['message']['content']
    result_reasoning = response['choices'][0]['message']['reasoning_content']
    return result,result_reasoning

    
def tencent_chat(question="你好,你是谁？",temperature=0,model="ds-r1"):
    url = 'https://api.lkeap.cloud.tencent.com/v1/chat/completions'
    apikey = "sk-7EMd6cPXZkcgPRBDJjLzU8hG5QFcZ3Eyux8fik5nZnBCNIJf"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    models ={
        'ds-r1':"deepseek-r1",
        'ds-v3_old':"deepseek-v3",
        'ds-v3':"deepseek-v3-0324",
    }
    data = {
        "model": models[model],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    result = response['choices'][0]['message']['content']
    result_reasoning = response['choices'][0]['message']['reasoning_content']
    return result,result_reasoning


async def tencent_chat_async(question="你好,你是谁？", temperature=0, model="ds-r1", timeout=1200):
    url = 'https://api.lkeap.cloud.tencent.com/v1/chat/completions'
    apikey = "sk-7EMd6cPXZkcgPRBDJjLzU8hG5QFcZ3Eyux8fik5nZnBCNIJf"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
    models ={
        'ds-r1':"deepseek-r1",
        'ds-v3_old':"deepseek-v3",
        'ds-v3':"deepseek-v3-0324",
    }
        
    data = {
        "model": models[model],
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
                if response.status == 500 or response.status == 429:
                    print(f"Error: HTTP {response.status} sleep 120s and retry")
                    await asyncio.sleep(120) # 120s
                    return await ark_chat_async(question, temperature, model, timeout)
                if response.status != 200:
                    print(f"Error: HTTP {response.status}")
                    return {'content': '', 'reasoning_content': ''}
                response_json = await response.json()
                result = response_json['choices'][0]['message']
                return result
    except asyncio.TimeoutError:
        print(f"Timeout error for question: {question[:50]}...")
        return {'content': '', 'reasoning_content': ''}
    except Exception as e:
        print(f"Error processing question: {question[:50]}... Error: {str(e)}")
        return {'content': '', 'reasoning_content': ''}





async def ark_chat_async(question="你好,你是谁？", temperature=0, model="ds-r1-7b_Batch", timeout=1200):
    url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
    url = "https://ark.cn-beijing.volces.com/api/v3/batch/chat/completions"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {apikey}'
    }
        
    data = {
        "model": ark_models_entrypoints[model],
        "stream": False,
        "messages": [
            # {
            #     "role": "system",
            #     "content": "You are a helpful assistant."
            # },
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
                if response.status == 500 or response.status == 429:
                    print(f"Error: HTTP {response.status} sleep 120s and retry")
                    await asyncio.sleep(120) # 120s
                    return await ark_chat_async(question, temperature, model, timeout)
                if response.status != 200:
                    print(f"Error: HTTP {response.status}")
                    return {'content': '', 'reasoning_content': ''}
                response_json = await response.json()
                result = response_json['choices'][0]['message']
                return result
    except asyncio.TimeoutError:
        print(f"Timeout error for question: {question[:50]}...")
        return {'content': '', 'reasoning_content': ''}
    except Exception as e:
        print(f"Error processing question: {question[:50]}... Error: {str(e)}")
        return {'content': '', 'reasoning_content': ''}



async def batch_ark_chat_async(questions: List[str], temperature=0, model="ds-r1-7b_Batch", timeout=1200) -> List[Tuple[str, str]]:
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
    return [{'content': '', 'reasoning_content': ''} if isinstance(r, Exception) else r for r in results]

async def main_async():
    questions = [
        "你是谁？",
        "你能做什么？",
        "498*874=?"
    ]
    # results = await batch_ark_chat_async(questions)
    results = await tencent_chat_async(questions,model="ds-r1")
    for i, (res) in enumerate(results):
        print(f"\nQuestion {i+1}: {questions[i]}")
        print(f"Response: {res['content']}")
        print(f"Reasoning: {res['reasoning_content']}\n\n")

if __name__ == "__main__":
    
    print("\nRunning asynchronous batch processing:")
    asyncio.run(main_async())  # test ok

    # #  test ollama
    # prompt = "666x333=?"
    # # res = llm_chat(prompt, model="qwen2.5:32b",temperature=0.5)
    # # print(res)

    # res = tencent_chat_async(prompt, model="ds-r1",temperature=0.5)
    # print(res)



