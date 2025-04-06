import pandas  as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def correct_in_steps(row):
    #  answer列和reasoning列中包含sum_answer
    # 检查是否有 这两列
    is_correct_in_steps = 0 
    sum_str = str(row['sum'])
    sum_str_comma = '{:,}'.format(int(sum_str))
    if 'answer' in row.index  and len(str(row['answer'])) > 0 :
        answer_str = str(row['answer'])
        if sum_str in answer_str:
            is_correct_in_steps = 1 
        # 可能答案中是 333,333 这种形式,需要给 sum_str 加逗号
        if sum_str_comma in answer_str:
            is_correct_in_steps = 1 
    if 'reasoning' in row.index  and len(str(row['reasoning'])) > 0 :
        reasoning_str = str(row['reasoning'])
        if sum_str in reasoning_str:
            is_correct_in_steps = 1 
        if sum_str_comma in reasoning_str:
            is_correct_in_steps = 1 
    return is_correct_in_steps

def diff_sort(row):
    diff_str = row['diff_str']
    
    temp = diff_str.split('\n')
    if len(temp) !=2:
        # 报错
        raise ValueError(f"diff_str {diff_str} is not valid")
    sum1 = temp[0].strip().split('-')
    sum2 = temp[1].strip().split('-')
    diff_part_count = 0
    diff_pos = 0
    
    for i in range(len(sum1)):
        if int(sum1[i]) == int(sum2[i]):
            continue
        diff_part_count +=1
        diff_pos = i
    if diff_part_count >1:
        # 多个片段不同，复杂类型
        return "value_nd"
    diff = int(sum1[diff_pos]) - int(sum2[diff_pos])
    if set(sum1[diff_pos]) != set(sum2[diff_pos]):
        if 0<diff <9:
            # 简单类型
            return "carry_1d"
        elif -9<diff <0:
            return "sub_1d"
        else:
            return "value_nd"
    # 两个字符串，只是顺序不同
    else:
        if len(sum1[diff_pos]) == 2:
            return "order_2d"
        else:
            return "value_nd"


fp = "data/eval_ds-r1_Batch_details_digits2-22.csv"
#fp = 'data/eval_ds-r1-7b_Batch_details_digits2-22.csv'
# fp =  'data/eval_ds-r1-32b_Batch_details_digits2-22.csv'
# fp = 'data/eval_ds-v3_Batch_details_digits2-22.csv'
# fp = 'data/eval_qwen2.5_details_digits2-22.csv'
# fp = "data/eval_doubao-pro_Batch_details_digits2-22.csv"


def error_sort(fp): 
    # if not exists, exit
    if not os.path.exists(fp):
        print(f"file {fp} not exists")
        exit()
    print(f"processing {fp}")
    df = pd.read_csv(fp)
    # correct ==1 的行数
    df['success'] = (df['correct'] == 1).astype(int)
    # df['error'] = (df['correct'] == 0).astype(int)
    df_part = df[['num_digits','success']]
    df_part = df_part.groupby(['num_digits']).sum()

    # 只分析correct ==0 的行
    df = df[df['correct'] == 0]
    # Apply the diff_sort function and create new columns
    df['error_type'] = df.apply(diff_sort, axis=1)
    # Create 4 new columns for each error type
    error_types = ['carry_1d', 'sub_1d', 'order_2d', 'value_nd']
    for error_type in error_types:
        df[error_type] = (df['error_type'] == error_type).astype(int)
    
    # 统计每组num_digits中 以下列之和，并生成csv
    columns = ['num_digits','carry_1d', 'sub_1d', 'order_2d', 'value_nd']
    df2 = df[columns]
    df2 = df2.groupby(['num_digits']).sum()
    # 合并df_correct和df2
    df2 = pd.concat([df_part, df2], axis=1)
    # 全部除2，换算成百分比,实际有200行
    df2 = df2.div(2)  
    df2 = df2.fillna(0)
    # 计算解析失败百分比 1- successs - 其他错误类型
    df2['parse_failed'] = 100 - df2['success'] - df2[error_types].sum(axis=1)
    return df2


def process_correct_in_steps(fp): 
    # if not exists, exit
    if not os.path.exists(fp):
        print(f"file {fp} not exists")
        exit()
    print(f"processing {fp}")
    df = pd.read_csv(fp)
    df = df[df['correct'] == 0]
    df['error'] = (df['correct'] == 0).astype(int)
    df['correct_in_steps'] = df.apply(correct_in_steps, axis=1)
    columns = ['num_digits','correct_in_steps','error']
    df2 = df[columns]
    df2 = df2.groupby(['num_digits']).sum()
    # 计算corrent_in_steps占error的百分比
    df2['correct_in_steps_pct%'] = df2['correct_in_steps'] / df2['error'] * 100
    return df2

df = error_sort(fp)
# df.to_csv("data_ark/eval_ds-r1-7b_Batch_details_digits2-22_error_sort.csv",index=False)

models = ["ds-r1-7b_Batch","ds-r1-32b_Batch","ds-v3_Batch","ds-r1_Batch","qwen2.5-ollama-q4","doubao-pro_Batch"]

def plot_error_analysis(df,model):
    for model in models:
        fp = f"data_ark/eval_{model}_details_digits2-22.csv"
        df = error_sort(fp)
        # 创建图表
        plt.figure(figsize=(12, 8))
        # 设置样式
        sns.set_style("whitegrid")
        # 绘制柱状图
        df.plot(kind='bar', stacked=True)
        plt.title(f'Error Analysis  {model}')
        plt.xlabel('Number of Digits')
        plt.ylabel('Percentage%')
        plt.legend(title='Error Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 保存图片
        outdir = "result"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        output_path = os.path.join(outdir, os.path.basename(fp).replace('.csv', '_error_analysis.png'))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")




for model in models:
    fp = f"data_ark/eval_{model}_details_digits2-22.csv"
    df = process_correct_in_steps(fp)
    print(df)






