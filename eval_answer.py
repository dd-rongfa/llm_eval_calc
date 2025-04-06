import re
import os
import csv
import time
import pandas as pd
import asyncio
from  llm import batch_ark_chat_async,llm_chat,batch_deepseek_chat_async,deepseek_chat,ollama_chat




def extract_boxed_answer(text):
    # 首先尝试匹配 \boxed{...} 格式（最优先）
    # 使用 re.DOTALL 标志来让 . 匹配换行符
    boxed_pattern = r"\\boxed\{(.*?)\}"
    boxed_matches = re.findall(boxed_pattern, text, re.DOTALL)
    if boxed_matches:
        answer = boxed_matches[-1].strip()
        # 处理 LaTeX 格式 - 使用字符串替换而非正则表达式
        answer = answer.replace("\\!", "")
        answer = answer.replace("\\,", "")
        answer = answer.replace("\\:", "")
        answer = answer.replace("\\;", "")
        # 移除所有空格
        answer = answer.replace(" ", "")
        # 移除所有逗号
        answer = answer.replace(",", "")
        return answer

#// 追加结果到csv

def write_result(result,filename='details.txt'):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(result)

# 记录异常的回复，无法解析的
def write_error_result(result,filename='error_details.txt'):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        f.write(result)
        f.write('\n\n\n')

    

def compare_num_str(num_str1, num_str2):
    if not num_str1 or not num_str2:
        return num_str1, num_str2
    # 反转字符串,从后往前比较   
    num_str1 = num_str1[::-1]
    num_str2 = num_str2[::-1]
     # 如果长度不同，补0
    if len(num_str1) < len(num_str2):
        num_str1 = num_str1 + '0' * (len(num_str2) - len(num_str1))
    elif len(num_str1) > len(num_str2):
        num_str2 = num_str2 + '0' * (len(num_str1) - len(num_str2))

    result1 = []
    result2 = []
    
    # 使用双指针，避免嵌套循环
    i = 0
    start = 0
    is_same = (num_str1[0] == num_str2[0])
    
    while i < min(len(num_str1), len(num_str2)):
        # 当前字符比较状态与之前不同时
        if (num_str1[i] == num_str2[i]) != is_same:
            result1.append(num_str1[start:i])
            result2.append(num_str2[start:i])
            start = i
            is_same = not is_same
        i += 1
    
    # 添加最后一个部分
    result1.append(num_str1[start:i])
    result2.append(num_str2[start:i])
    return '-'.join(result1)[::-1], '-'.join(result2)[::-1]


async def eval_answer_ark(sample_file,model="ds-r1-32b_Batch",start_num_digits=2,end_num_digits=22,if_only_even=True):
    print(f"eval_answer_ark {model} {start_num_digits}-{end_num_digits}")
    data_dir = "./data_ark/"
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{model}_summary_digits{start_num_digits}-{end_num_digits}.csv')
    result_file = os.path.join(data_dir, f'eval_{model}_details_digits{start_num_digits}-{end_num_digits}.csv')
    error_file = os.path.join(data_dir, f'eval_{model}_error_details_digits{start_num_digits}-{end_num_digits}.txt')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%','cost_time','tokens_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        if if_only_even and num_digits %2 != 0:
            continue
        if  num_digits < start_num_digits or num_digits > end_num_digits:   
            continue
        df_num_digits = df[df.iloc[:,0] == num_digits]
        calc_error_count = 0
        parse_error_count = 0
        correct_count = 0
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        batch_size = 100
        for i in range(0,len(questions),batch_size):
            results.extend(await batch_ark_chat_async(questions[i:i+batch_size],model=model,timeout=3000))

        for index,result in enumerate(results):
            if result is None:
                parse_error_count += 1
                print("network error")
                continue
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result['choices'][0]['message'].get('content','')
            reasoning = result['choices'][0]['message'].get('reasoning_content','')
            token_count += result['usage']['total_tokens']
            answer_only = extract_boxed_answer(result_raw)
            if answer_only is None:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue    
            try:
                answer_only = int(answer_only)
            except:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue
            if answer_only == int(num_sum):
                correct_count += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                calc_error_count  += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        total = correct_count + calc_error_count + parse_error_count 
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100:.2f}, calc error%: {calc_error_count/total*100:.2f}, parse_error%: {parse_error_count/total*100:.2f} cost_time: {time.time() - start_time:.2f}s tokens_count: {token_count}")
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}",f"{time.time() - start_time:.2f}s",token_count],summary_file)



def eval_answer_ds(sample_file,model="ds-r1",start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000):
    print(f"eval_answer_ds {model} {start_num_digits}-{end_num_digits}")
    data_dir = "./data/"
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{model}_summary_digits{start_num_digits}-{end_num_digits}.csv')
    result_file = os.path.join(data_dir, f'eval_{model}_details_digits{start_num_digits}-{end_num_digits}.csv')
    error_file = os.path.join(data_dir, f'eval_{model}_error_details_digits{start_num_digits}-{end_num_digits}.txt')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%','cost_time','tokens_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        if if_only_even and num_digits %2 != 0:
            continue
        if  num_digits < start_num_digits or num_digits > end_num_digits:   
            continue
        df_num_digits = df[df.iloc[:,0] == num_digits]
        calc_error_count = 0
        parse_error_count = 0
        correct_count = 0
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        batch_size = 100
        for i in range(0,len(questions)):
            response = deepseek_chat(questions[i],model=model,timeout=timeout)
            # 等待20s 后重试
            if response is None:
                print(f"question {i} of {num_digits}: {questions[i]} timeout, sleep 300s and retry")
                time.sleep(300)
                response = deepseek_chat(questions[i],model=model,timeout=timeout)
            print(f"question {i} of {num_digits}: {questions[i]} response: {response.get('id') if response else 'timeout'}" )
            results.append(response)

        for index,result in enumerate(results):
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result['choices'][0]['message'].get('content','')
            reasoning = result['choices'][0]['message'].get('reasoning_content','')
            token_count += result['usage']['total_tokens']
            answer_only = extract_boxed_answer(result_raw)
            if answer_only is None:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue    
            try:
                answer_only = int(answer_only)
            except:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue
            if answer_only == int(num_sum):
                correct_count += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                calc_error_count  += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        total = correct_count + calc_error_count + parse_error_count 
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100:.2f}, calc error%: {calc_error_count/total*100:.2f}, parse_error%: {parse_error_count/total*100:.2f} cost_time: {time.time() - start_time:.2f}s tokens_count: {token_count}")
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}",f"{time.time() - start_time:.2f}s",token_count],summary_file)



async def eval_answer_ds_async(sample_file,model="ds-r1",start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000):
    print(f"eval_answer_ds_async {model} {start_num_digits}-{end_num_digits}")
    data_dir = "./data_ark/"
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{model}_summary_digits{start_num_digits}-{end_num_digits}.csv')
    result_file = os.path.join(data_dir, f'eval_{model}_details_digits{start_num_digits}-{end_num_digits}.csv')
    error_file = os.path.join(data_dir, f'eval_{model}_error_details_digits{start_num_digits}-{end_num_digits}.txt')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%','cost_time','tokens_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        if if_only_even and num_digits %2 != 0:
            continue
        if  num_digits < start_num_digits or num_digits > end_num_digits:   
            continue
        df_num_digits = df[df.iloc[:,0] == num_digits]
        calc_error_count = 0
        parse_error_count = 0
        correct_count = 0
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        batch_size = 10
        for i in range(0,len(questions),batch_size):
            results.extend(await batch_deepseek_chat_async(questions[i:i+batch_size],model=model,timeout=timeout))
        for index,result in enumerate(results):
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result['choices'][0]['message'].get('content','')
            reasoning = result['choices'][0]['message'].get('reasoning_content','')
            token_count += result['usage']['total_tokens']
            answer_only = extract_boxed_answer(result_raw)
            if answer_only is None:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue    
            try:
                answer_only = int(answer_only)
            except:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue
            if answer_only == int(num_sum):
                correct_count += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                calc_error_count  += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        total = correct_count + calc_error_count + parse_error_count 
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100:.2f}, calc error%: {calc_error_count/total*100:.2f}, parse_error%: {parse_error_count/total*100:.2f} cost_time: {time.time() - start_time:.2f}s tokens_count: {token_count}")
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}",f"{time.time() - start_time:.2f}s",token_count],summary_file)


def eval_answer_ollama(data_dir,sample_file,model="qwen2.5",start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000):
    print(f"eval_answer_ollama {model} {start_num_digits}-{end_num_digits}")
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{model}_summary_digits{start_num_digits}-{end_num_digits}.csv')
    result_file = os.path.join(data_dir, f'eval_{model}_details_digits{start_num_digits}-{end_num_digits}.csv')
    error_file = os.path.join(data_dir, f'eval_{model}_error_details_digits{start_num_digits}-{end_num_digits}.txt')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%','cost_time','tokens_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        if if_only_even and num_digits %2 != 0:
            continue
        if  num_digits < start_num_digits or num_digits > end_num_digits:   
            continue
        df_num_digits = df[df.iloc[:,0] == num_digits]
        calc_error_count = 0
        parse_error_count = 0
        correct_count = 0
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        # batch_size = 1
        for i in range(0,len(questions)):
            results.append(ollama_chat(questions[i],model=model,timeout=timeout))
        for index,result in enumerate(results):
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result['message'].get('content','')
            # 提取 字符中的<thinking>...</thinking> 
            reasoning = re.findall(r'<thinking>(.*?)</thinking>', result_raw, re.DOTALL)
            reasoning = reasoning[0] if reasoning else ''
            token_count += result['eval_count'] + result['prompt_eval_count']
            answer_only = extract_boxed_answer(result_raw)
            if answer_only is None:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue    
            try:
                answer_only = int(answer_only)
            except:
                parse_error_count += 1
                write_error_result(f"question {index} of {num_digits}: {question}\nreasoning: {reasoning}\nerror result: {result_raw}",error_file)
                continue
            if answer_only == int(num_sum):
                correct_count += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                calc_error_count  += 1
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        total = correct_count + calc_error_count + parse_error_count 
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100:.2f}, calc error%: {calc_error_count/total*100:.2f}, parse_error%: {parse_error_count/total*100:.2f} cost_time: {time.time() - start_time:.2f}s tokens_count: {token_count}")
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}",f"{time.time() - start_time:.2f}s",token_count],summary_file)


if __name__ == "__main__":
    model = "ds-r1_Batch"
    sample_file = os.path.join('data', "sample_questions200_addnocarry_digits2-30.csv")
    # asyncio.run(eval_answer_ark(sample_file,model,start_num_digits=2,end_num_digits=22,if_only_even=True))
    # asyncio.run(eval_answer_ds_async(sample_file,model,start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000))
    # model = "ds-v3"
    # asyncio.run(eval_answer_ds_async(sample_file,model,start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000))

    eval_answer_ollama(data_dir="./data_ollama/",sample_file=sample_file,model="qwen2.5",start_num_digits=2,end_num_digits=22,if_only_even=True,timeout=3000)


