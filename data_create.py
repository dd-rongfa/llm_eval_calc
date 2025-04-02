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
    is_carry = False
    for i in range(1,num_digits):
        a = a*10 + num_1[i]
        b = b*10 + num_2[i]
        if num_1[i] + num_2[i] >= 10:
            is_carry = True
    data.append(num_digits) 
    data.append(1 if is_carry else 0)       
    data.append(a)
    data.append(b)
    data.append(a+b)
    return is_carry,data



def generate_add_no_carry_data(num_samples,num_start,num_end):
    start_time = time.time()
    data_list = []
    for num_digits in range(num_start,num_end):
        for i in range(num_samples):
            is_carry,data = generate_add_no_carry_data_fixed_length(num_digits=num_digits)
            if is_carry:
                print(f"error data: {data} is carry ed")
            else:   
                data_list.append(data)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # // 写到csv
    import csv
    file_name = f"add_no_carry_data_samples_{num_samples}_digits_{num_start}_{num_end}_random_seed_{random_seed}.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num_digits', 'is_carry','num1', 'num2', 'sum'])
        for data in data_list:
            writer.writerow(data)

# 如何产生连续进位的数据

# 如何产生混合运算的数据


if __name__ == "__main__":
    num_samples = 200
    num_start = 2
    num_end = 22
    generate_add_no_carry_data(num_samples,num_start,num_end)

