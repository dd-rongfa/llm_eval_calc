import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt



models = ["qwen2.5",'ds-r1-7b_Batch','ds-r1-32b_Batch','ds-r1_Batch','ds-v3_Batch','doubao-pro_Batch']

def accuracy_show(models):
    df_all = []
    for model in models:
        fp = f"data_ark/eval_{model}_summary_digits2-22.csv"
        if not os.path.exists(fp):
            print(f"file {fp} not exists")
            continue
        
        df = pd.read_csv(fp)[['num_digits',"correct%","calc error%"]]
        df['model'] = model  # Add model column
        df_all.append(df)

    # Combine all dataframes
    combined_df = pd.concat(df_all, ignore_index=True)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot line for each model
    for model in models:
        model_data = combined_df[combined_df['model'] == model]
        plt.plot(model_data['num_digits'], model_data['correct%'], marker='o', label=model)
    # 设定x 刻度是2，4，6，8，10，12，14，16，18，20
    plt.xticks(np.arange(2, 24, 2))
    plt.xlabel('Number of Digits')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs Number of Digits')
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs('result', exist_ok=True)
    plt.savefig('result/accuracy_comparison_digits2-22.png')
    #plt.show()  
    plt.close()



if __name__ == "__main__":
    accuracy_show(models)






