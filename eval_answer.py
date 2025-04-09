import re
import os
import csv
import time
import pandas as pd
import asyncio
import json
import tqdm

from utils import chat_llm_batch,async_chat_llm_batch

# 提取boxed答案
def extract_boxed_answer_int(text):
    # 首先尝试匹配 \boxed{...} 格式（最优先）
    # 使用 re.DOTALL 标志来让 . 匹配换行符
    boxed_pattern = r"\\boxed\{(.*?)\}"
    boxed_matches = re.findall(boxed_pattern, text, re.DOTALL)
    if not boxed_matches:
        return None
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
    try:
        return int(answer)
    except:
        return None

#// 追加结果到csv
def write_result(result,filename='details.txt',mode='a'):
    with open(filename, 'a' if mode == 'a' else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(result)


# 比较两个数字字符串,分割位相同部分和不同部分，方便看出错误的位置
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


def eval_answers(data_dir,sample_file,start=2,end=22,only_even=True,model_config=None,mode='a'):
    '''
    评估答案
    mode: a 追加,只测漏检的 ，w 覆盖，重测所有
    '''
    model = model_config['model_name']
    server = model_config['server']
    print(f"eval_answer_ollama {model} {start}-{end}")
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{server}_{model}_summary_digits{start}-{end}.csv')
    result_file = os.path.join(data_dir, f'eval_{server}_{model}_details_digits{start}-{end}.csv')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%']
    write_result(summary_header,summary_file,mode='w')
    if mode != 'a' or not os.path.exists(result_file):
        write_result(header,result_file,mode='w') 

    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    # 根据范围筛选
    num_digits_list = [num_digits for num_digits in num_digits_list if num_digits >= start and num_digits <= end]
    # 根据是否只偶数筛选
    if only_even:
        num_digits_list = [num_digits for num_digits in num_digits_list if num_digits %2 == 0]
    
    previous_num_digits = []
    if mode == 'a':
        # 追加测试，之前没测试完的，丢失的，补测
        # 获取之前测试过的num_digits
        with open(result_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过第一行
            next(reader)
            for row in reader:
                previous_num_digits.append((int(row[0]), int(row[1]), int(row[2])))
            
    print(f"num_digits_list length: {len(num_digits_list)} {num_digits_list}")
    df = df[df.iloc[:,0].isin(num_digits_list)]

    # 遍历num_digits_list
    for num_digits in tqdm.tqdm(num_digits_list,desc=f"eval {server} {model} {start}-{end}"):
        df_num_digits = df[df.iloc[:,0] == num_digits]
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            # 之前测试过的，跳过
            if (int(row.iloc[0]),int(row.iloc[1]),int(row.iloc[2])) in previous_num_digits:
                # print(f"skip {row[0]} {row[1]} {row[2]}")
                continue
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        batch_size = 5
        for i in tqdm.tqdm(range(0,len(questions),batch_size), desc=f"Processing batches for {num_digits} digits", leave=False):
            results.extend(chat_llm_batch(questions[i:i+batch_size],model_config))
        for index,result in enumerate(results):
            if result is None:
                # print("网络异常")
                continue
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result.get('content','')
            reasoning = result.get('reasoning_content','')
            token_count += result.get('total_tokens',0)
            answer_only = extract_boxed_answer_int(result_raw)
            if answer_only is None:
                diff = 0
                diff_str = ""
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'parse_error'],result_file)
                continue    
            if answer_only == int(num_sum):
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        df_temp = pd.read_csv(result_file)
        df_temp = df_temp[df_temp.iloc[:,0] == num_digits]
        df_temp['parse_error'] = df_temp.iloc[:,-1].apply(lambda x: 1 if x == 'parse_error' else 0)
        df_temp['error'] = df_temp.apply(lambda row: 1 if row['correct'] == 0 and row['parse_error'] == 0 else 0,axis=1)
        total = df_temp.shape[0]
        correct_count = df_temp['correct'].sum()
        calc_error_count = df_temp['error'].sum()
        parse_error_count = df_temp['parse_error'].sum()
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}"],summary_file)



async def async_eval_answers(data_dir,sample_file,start=2,end=22,only_even=True,model_config=None,mode='a'):
    '''
    评估答案
    mode: a 追加,只测漏检的 ，w 覆盖，重测所有
    '''
    model = model_config['model_name']
    server = model_config['server']
    print(f"eval_answer_ollama {model} {start}-{end}")
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f'eval_{server}_{model}_summary_digits{start}-{end}.csv')
    result_file = os.path.join(data_dir, f'eval_{server}_{model}_details_digits{start}-{end}.csv')
    header = ['num_digits', 'num1', 'num2', 'sum',"answer_only",'diff','diff_str','correct','question','answer','reasoning','comment',]
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%']
    write_result(summary_header,summary_file,mode='w')
    
    if mode != 'a' or not os.path.exists(result_file):
        write_result(header,result_file,mode='w') 
        
    # 如果文件不在，就报错
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"sample_file {sample_file} not found")
    # num_digits,num1,num2,sum,question_en,question
    df = pd.read_csv(sample_file)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    # 根据范围筛选
    num_digits_list = [num_digits for num_digits in num_digits_list if num_digits >= start and num_digits <= end]
    # 根据是否只偶数筛选
    if only_even:
        num_digits_list = [num_digits for num_digits in num_digits_list if num_digits %2 == 0]
    
    previous_num_digits = []
    if mode == 'a':
        # 追加测试，之前没测试完的，丢失的，补测
        # 获取之前测试过的num_digits
        with open(result_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过第一行
            next(reader)
            for row in reader:
                previous_num_digits.append((int(row[0]), int(row[1]), int(row[2])))
            
    print(f"num_digits_list length: {len(num_digits_list)} {num_digits_list}")
    df = df[df.iloc[:,0].isin(num_digits_list)]

    # 遍历num_digits_list
    for num_digits in tqdm.tqdm(num_digits_list,desc=f"eval {server} {model} {start}-{end}"):
        df_num_digits = df[df.iloc[:,0] == num_digits]
        start_time = time.time()
        token_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            # 之前测试过的，跳过
            if (int(row.iloc[0]),int(row.iloc[1]),int(row.iloc[2])) in previous_num_digits:
                # print(f"skip {row[0]} {row[1]} {row[2]}")
                continue
            question = row.iloc[4]
            questions.append(question)
        # 一次最多20个
        results = []
        batch_size = 5
        for i in tqdm.tqdm(range(0,len(questions),batch_size), desc=f"Processing batches for {num_digits} digits", leave=False):
            results.extend(await async_chat_llm_batch(questions[i:i+batch_size],model_config))
        for index,result in enumerate(results):
            if result is None:
                # print("网络异常")
                continue
            num_digits,num1,num2,num_sum,question,question_en = df_num_digits.iloc[index]
            result_raw = result.get('content','')
            reasoning = result.get('reasoning_content','')
            token_count += result.get('total_tokens',0)
            answer_only = extract_boxed_answer_int(result_raw)
            if answer_only is None:
                diff = 0
                diff_str = ""
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'parse_error'],result_file)
                continue    
            if answer_only == int(num_sum):
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,1,questions[index],result_raw,reasoning,''],result_file)
            else:
                diff = answer_only - int(num_sum)
                diff_str = "\n".join(compare_num_str(str(answer_only),str(num_sum)))
                write_result([num_digits,num1,num2,num_sum,answer_only,diff,diff_str,0,questions[index],result_raw,reasoning,'error'],result_file)
            # 记录问题和答案
        df_temp = pd.read_csv(result_file)
        df_temp = df_temp[df_temp.iloc[:,0] == num_digits]
        df_temp['parse_error'] = df_temp.iloc[:,-1].apply(lambda x: 1 if x == 'parse_error' else 0)
        df_temp['error'] = df_temp.apply(lambda row: 1 if row['correct'] == 0 and row['parse_error'] == 0 else 0,axis=1)
        total = df_temp.shape[0]
        correct_count = df_temp['correct'].sum()
        calc_error_count = df_temp['error'].sum()
        parse_error_count = df_temp['parse_error'].sum()
        write_result([num_digits,total,f"{correct_count/total*100:.2f}",f"{calc_error_count/total*100:.2f}",f"{parse_error_count/total*100:.2f}"],summary_file)


if __name__ == "__main__":
    sample_file = os.path.join('data', "sample_questions10_addnocarry_digits2-30.csv")
    config = {
        'server': 'deepseek',
        "model_name": "ds-v3",
        "timeout": 3000,
        "temperature": 0.6,
        'max_tokens': 8192,

    }   
    ## test ok
    # asyncio.run(async_eval_answers(data_dir="./data_test/",sample_file=sample_file,model_config=model_config,mode='w'))
    ## test ok
    eval_answers(data_dir="./data_test/",sample_file=sample_file,start=2,end=42 ,model_config=config)


