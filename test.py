from openai import OpenAI

# client = OpenAI(
#     base_url='http://localhost:11434/v1/',

#     # required but ignored
#     api_key='ollama',
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             'role': 'user',
#             'content': '你是谁？',
#         }
#     ],
#     model='qwen2.5:latest',
# )

# print(chat_completion.choices[0].message.content)

from env import apikey_ark, apikey_tencent


import asyncio
from openai import AsyncOpenAI

# 替换为你的OpenAI API密钥
client = AsyncOpenAI(api_key="ollama")
client.base_url = "http://localhost:11434/v1/"
# ark
base_url = "https://ark.cn-beijing.volces.com/api/v3/batch"
model = "ep-bi-20250401151117-zwvf8"


async def async_ask_gpt(prompt: str, model: str,url:str,api_key:str, max_tokens: int = 8192,timeout: int = 1800) -> dict:
    try:
        client.base_url = url 
        client.api_key = api_key
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=max_tokens,
            timeout=timeout
        )
        return response.json()
    except Exception as e:
        print(f"error: {e}")
        return None

async def aysnc_chat_batch(questions:list):
    # 定义两个不同的提示
    prompt1 = "你是谁？"
    prompt2 = "3+3=?"
    
    # 同时发起两个请求
    task1 = async_ask_gpt(prompt1,model,base_url,apikey_ark)
    task2 = async_ask_gpt(prompt2,model,base_url,apikey_ark)
    
    # 等待两个任务完成
    responses = await asyncio.gather(task1, task2)
    
    # 打印结果
    print(f"问题1: {prompt1}")
    print(f"回答1: {responses[0]}\n")
    
    print(f"问题2: {prompt2}")
    print(f"回答2: {responses[1]}\n")

if __name__ == "__main__":
    asyncio.run(main())



# client = OpenAI(
#     api_key = os.environ.get("ARK_API_KEY"),
#     base_url = "https://ark.cn-beijing.volces.com/api/v3",
# )

# # Non-streaming:
# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model = "ep-bi-20250401151117-zwvf8",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)
