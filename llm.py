import requests
import json
import asyncio
import aiohttp
from typing import List, Tuple

ark_models_entrypoints= {
    'ds-r1-7b':"ep-20250218161643-vl25s",
    'ds-r1-7b_Batch':"ep-bi-20250401151011-nzp5d",
    'ds-r1-32b':"ep-20250218161612-h68p4",
    'ds-v3': "ep-20250214165420-s2s6s",
    'ds-r1': "ep-20250214165338-52bjm",
    'doubao-lite':"ep-20241217204945-k7pmp",
    'doubao-pro':"ep-20241012144130-85pt2",
}

ark_url = "https://api.ark.com/v1/chat/completions"
apikey = "d0a38cca-7ef4-46ff-bda6-6433191af8eb"



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




# question ="你是谁？"
# res,res_reasoning = ark_chat(question=question)
# print(res)
# print(res_reasoning)



async def ark_chat_async(question="你好,你是谁？", temperature=0, model="ds-r1-7b_Batch", timeout=600):
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
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
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



async def batch_ark_chat_async(questions: List[str], temperature=0, model="ds-r1-7b_Batch", timeout=600) -> List[Tuple[str, str]]:
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
        "今天天气怎么样？",
        "498*874=?"
    ]
    results = await batch_ark_chat_async(questions)
    for i, (res) in enumerate(results):
        print(f"\nQuestion {i+1}: {questions[i]}")
        print(f"Response: {res['content']}")
        print(f"Reasoning: {res['reasoning_content']}\n\n")

if __name__ == "__main__":
    
    print("\nRunning asynchronous batch processing:")
    asyncio.run(main_async())  # test ok


