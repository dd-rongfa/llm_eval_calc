import requests
import json
import ollama
import re
import os
import csv
import time
import pandas as pd
import asyncio
from  llm import batch_ark_chat_async



def extract_boxed_answer(text):
    # 匹配 \boxed{...} 或 **...** 或 [答案] 等常见格式
    patterns = [
        r"\\boxed\{(.*?)\}",      # LaTeX 格式
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 返回最后一个匹配项（假设最终答案在末尾）
            return matches[-1].strip()
    
    # 如果没有匹配到，尝试提取最后一个纯数字（备用方案）
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[-1]
    
    return None


#// 追加结果到csv

def write_result(result,filename='details.txt'):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(result)


def eval_answer():
    sample_file = "add_no_carry_data_samples_1000_digits_1_30_random_seed_42.csv"
    summary_file = 'summary.csv'
    result_file = 'result_add_no_carry_data_samples_1000_digits_1_30_random_seed_42.csv'
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
    data_dir = "./experiment_add_no_carry"
    sample_file = os.path.join(data_dir, "add_no_carry_data_samples_200_digits_1_22_random_seed_42.csv")
    summary_file = os.path.join(data_dir, 'ark_summary_add_no_carry_data_samples_200_digits_1_22_random_seed_42.csv')
    result_file = os.path.join(data_dir, 'ark_result_add_no_carry_data_samples_200_digits_1_22_random_seed_42.csv')
    header = ['num_digits', 'is_carry','num1', 'num2', 'sum','diff','correct','question','answer','reasoning','comment','parse_error','responese_error']
    summary_header = ['num_digits', 'total_count','correct%','error%','parse_error%','responese_error%','cost_time','token_input_count','token_output_count']
    write_result(header,result_file) 
    write_result(summary_header,summary_file)
    token_input_count=0
    token_output_count=0
    df = pd.read_csv(sample_file,skiprows=1)
    # 获取第一列的所有可能
    num_digits_list = df.iloc[:,0].unique()
    for num_digits in num_digits_list:
        df_num_digits = df[df.iloc[:,0] == num_digits]
        error_count = 0
        parse_error_count = 0
        responese_error_count = 0
        correct_count = 0
        start_time = time.time()
        # 添加prompt，生成批量问题
        questions = []
        for index,row in df_num_digits.iterrows():
            question = f"请准确计算 {row[2]}+{row[3]}，把最终计算结果放 \\boxed{{}} 中，例如计算结果为2, 写成 \\boxed{{2}}"
            token_input_count += len(question)
            questions.append(question)
        
        # 一次最多20个
        results = []
        batch_size = 20
        for i in range(0,len(questions),batch_size):
            results.extend(await batch_ark_chat_async(questions[i:i+batch_size]))

        for index,result in enumerate(results):
            num_digits,is_carry,num1,num2,num_sum = df_num_digits.iloc[index]
            result_raw = result['content']
            reasoning = result['reasoning_content']
            token_output_count += len(result_raw) + len(reasoning)
            answer_only = extract_boxed_answer(result_raw)
            
            error_comment = ''
            flag = 0
            if answer_only is None:
                print(f"error result: {result_raw}")
                responese_error_count += 1
                error_comment = 'responese error'
            else:
                try:
                    answer_only = int(answer_only)
                except:
                    print(f"parse error: {result_raw}")
                    parse_error_count += 1
                    error_comment = 'parse error'
                if answer_only == int(num_sum):
                    correct_count += 1
                    flag = 1
                else:
                    error_count += 1
                    error_comment = 'answer error'
                    flag = 0
            # 记录问题和答案
            
            write_result([num_digits,is_carry,num1,num2,num_sum,answer_only-int(num_sum),flag,questions[index],result_raw,reasoning,error_comment,parse_error_count,responese_error_count],result_file)
        total = correct_count + error_count
        print(f"num_digits: {num_digits}, total: {total}, correct%: {correct_count/total*100}, error%: {error_count/total*100} parse_error%: {parse_error_count/total*100} responese_error%: {responese_error_count/total*100} cost_time: {time.time() - start_time} token_input_count: {token_input_count} token_output_count: {token_output_count}")
        write_result([num_digits,total,correct_count/total*100,error_count/total*100,parse_error_count/total*100,responese_error_count/total*100,time.time() - start_time,token_input_count,token_output_count],summary_file)

if __name__ == "__main__":
    asyncio.run(eval_answer_ark())



