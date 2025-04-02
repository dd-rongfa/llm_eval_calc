import requests
import json
import ollama
import re
import os
import csv
import time
import pandas as pd
import asyncio
from  llm import batch_ark_chat_async,llm_chat




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


def eval_answer():
    data_dir = "./experiment_add_no_carry"
    sample_file = os.path.join(data_dir, "add_no_carry_data_samples_500_digits_1_22_random_seed_42.csv")
    summary_file = os.path.join(data_dir, 'summary_add_no_carry_data_samples_500_digits_1_22_random_seed_42.csv')
    result_file = os.path.join(data_dir, 'result_add_no_carry_data_samples_500_digits_1_22_random_seed_42.csv')
    header = ['num_digits', 'is_carry','num1', 'num2', 'sum','result','correct','question','answer','comment']
    summary_header = ['num_digits', 'total_count','correct%','error%','cost_time']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    error_count = 0
    correct_count = 0
    prev_num_digits = 1
    start_time = time.time()
    model = "qwen2.5:latest"
    with open(sample_file, 'r') as f:
        reader = csv.reader(f)
        # 跳过表头第一行
        next(reader)
        for row in reader:
            num_digits = int(row[0])
            is_carry = int(row[1])
            num1 = int(row[2])
            num2 = int(row[3])
            num_sum = int(row[4])
            num_str= f"{num1}+{num2}"
            quetion = f"请准确计算 {num_str}，把最终计算结果放 \\boxed{{}} 中，例如计算结果为2, 写成 \\boxed{{2}}"
            result_raw = llm_chat(quetion,temperature=0,model=model)
            result = extract_boxed_answer(result_raw)
            if result is None:
                print(f"error result: {result_raw}")
                error_count += 1
                write_result([num_digits,is_carry,num1,num2,sum,'',0,quetion,result_raw,'responese error'],result_file)
                continue
            try:
                result = int(result)
            except:
                print(f"error result: {result_raw}")
                error_count += 1
                write_result([num_digits,is_carry,num1,num2,sum,'',0,quetion,result_raw,'responese error'],result_file)
                continue
            if result == num_sum:
                correct_count += 1
                write_result([num_digits,is_carry,num1,num2,num_sum,result,1,quetion,result_raw,''],result_file)
            else:
                error_count += 1
                write_result([num_digits,is_carry,num1,num2,num_sum,result,0,quetion,result_raw,'error'],result_file)
            # 更新统计
            if num_digits != prev_num_digits:
                cost_time = time.time() - start_time
                write_result([num_digits,correct_count+error_count,correct_count/(correct_count+error_count)*100,error_count/(correct_count+error_count)*100,cost_time],summary_file)
                prev_num_digits = num_digits
                correct_count = 0
                error_count = 0
                start_time = time.time()
            if (correct_count +error_count) % 100 == 0:
                print(f"num_digits: {num_digits}, correct_count: {correct_count}, error_count: {error_count} cost_time: {time.time() - start_time}")
            
                

async def eval_answer_ark():
    data_dir = "./data/ark_dsv3"
    model = "ds-v3_Batch"
    os.makedirs(data_dir, exist_ok=True)
    sample_file = os.path.join("add_no_carry_data_samples_500_digits_1_22_random_seed_42.csv")
    summary_file = os.path.join(data_dir, 'ark_dsv3_summary.csv')
    result_file = os.path.join(data_dir, 'ark_dsv3_details.csv')
    header = ['num_digits', 'is_carry','num1', 'num2', 'sum','diff','correct','question','answer','reasoning','comment','parse_error','responese_error']
    summary_header = ['num_digits', 'total_count','correct%','calc error%','parse_error%','responese_error%','cost_time','token_input_count','token_output_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    df = pd.read_csv(sample_file,skiprows=1)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        if num_digits %2 != 0:
            continue
        df_num_digits = df[df.iloc[:,0] == num_digits]
        calc_error_count = 0
        parse_error_count = 0
        responese_error_count = 0
        correct_count = 0
        start_time = time.time()
        token_input_count=0
        token_output_count=0
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = f"请正确计算 {row[2]}+{row[3]}，把最终计算结果放 \\boxed{{}} 中，例如计算结果为2, 写成 \\boxed{{2}}"
            token_input_count += len(question)
            questions.append(question)
        
        # 一次最多20个
        results = []
        batch_size = 100
        for i in range(0,len(questions),batch_size):
            results.extend(await batch_ark_chat_async(questions[i:i+batch_size],model=model))

        for index,result in enumerate(results):
            num_digits,is_carry,num1,num2,num_sum = df_num_digits.iloc[index]
            result_raw = result.get('content','')
            reasoning = result.get('reasoning_content','')
            token_output_count += len(result_raw) + len(reasoning)
            answer_only = extract_boxed_answer(result_raw)
            
            error_comment = ''
            flag = 0
            if answer_only is None:
                print(f"error result: {result_raw}")
                responese_error_count += 1
                error_comment = 'responese error'
                write_result([num_digits,is_carry,num1,num2,num_sum,0,0,questions[index],result_raw,reasoning,error_comment,parse_error_count,responese_error_count],result_file)
                continue
            try:
                answer_only = int(answer_only)
            except:
                print(f"parse error: {result_raw}")
                parse_error_count += 1
                error_comment = 'parse error'
                write_result([num_digits,is_carry,num1,num2,num_sum,0,0,questions[index],result_raw,reasoning,error_comment,parse_error_count,responese_error_count],result_file)
                continue
            if answer_only == int(num_sum):
                correct_count += 1
                flag = 1
            else:
                calc_error_count  += 1
                error_comment = 'answer error'
                flag = 0
            # 记录问题和答案
            write_result([num_digits,is_carry,num1,num2,num_sum,answer_only-int(num_sum),flag,questions[index],result_raw,reasoning,error_comment,parse_error_count,responese_error_count],result_file)
        total = correct_count + calc_error_count + parse_error_count + responese_error_count
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100}, calc error%: {calc_error_count/total*100}, parse_error%: {parse_error_count/total*100} responese_error%: {responese_error_count/total*100} cost_time: {time.time() - start_time} token_input_count: {token_input_count} token_output_count: {token_output_count}")
        write_result([num_digits,total,correct_count/total*100,calc_error_count/total*100,parse_error_count/total*100,responese_error_count/total*100,time.time() - start_time,token_input_count,token_output_count],summary_file)

if __name__ == "__main__":
    asyncio.run(eval_answer_ark())



