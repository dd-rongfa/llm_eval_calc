# 构造数据集脚本

'''
测试样例设计：
- 基础组（无进位简单计算）
- 压力测试组（连续进位）
- 混合运算组（含多个运算符的复合表达式）

'''
import time
import random
# 设定随机数seed
random_seed = 42
random.seed(random_seed)

# 如何产生完全不进位的数据

# 固定长度的数字,无进位的加法，单个样本生成
def generate_add_no_carry_data_fixed_length(num_digits=4):
    data = []
    # 1 个 1-9,3个0-9
    # 第一位 1-8，后三位 0-9
    num_1 = [random.randint(1, 8)] # 第一位 1-8
    for i in range(1,num_digits):
        num_1.append(random.randint(0, 9))
    # 第一个数字，只能根据第一个数字选剩下的不会生成进位的空间
    num_2 =[random.randint(1,9-num_1[0])] 
    for i in range(1,num_digits):
        num_2.append(random.randint(0,9-num_1[i]))
    a,b = num_1[0],num_2[0]
    for i in range(1,num_digits):
        a = a*10 + num_1[i]
        b = b*10 + num_2[i]
    data.append(num_digits) 
    data.append(a)
    data.append(b)
    data.append(a+b)
    return data


def generate_add_no_carry_data(num_samples,num_start,num_end,file_name):
    start_time = time.time()
    data_list = []
    for num_digits in range(num_start,num_end):
        for i in range(num_samples):
            data = generate_add_no_carry_data_fixed_length(num_digits=num_digits)
            question = f"列竖式计算表达式的值，计算结果放 \\boxed{{}} 中，例如结果为2, 写成 \\boxed{{2}}。表达式:{data[1]}+{data[2]}"
            question_en = f"Perform vertical addition to calculate the value of the expression, place the result inside \\boxed{{}}. For example, if the result is 2, write it as \\boxed{{2}}. Expression: {data[1]} + {data[2]}"
            data.append(question)
            data.append(question_en)
            data_list.append(data)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # // 写到csv
    import csv
    with open(file_name, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['digits', 'num1', 'num2', 'sum','question','question_en'])
        for data in data_list:
            writer.writerow(data)



if __name__ == "__main__":
    import os
    num_samples = 200
    num_start = 2
    num_end = 31
    fdir = "./data/"
    os.makedirs(fdir, exist_ok=True)
    file_name = os.path.join(fdir, f"sample_questions{num_samples}_addnocarry_digits{num_start}-{num_end-1}.csv")
    generate_add_no_carry_data(num_samples,num_start,num_end,file_name)

