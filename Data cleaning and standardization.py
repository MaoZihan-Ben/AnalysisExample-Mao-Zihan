import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handle_missing_values(df, filename):
    """处理缺失值并生成缺失值热图"""
    original_rows = len(df)
    df_dropped = df.dropna() 
    missing_rows = original_rows - len(df_dropped)
    
    print(f"\n===== 处理 {filename} 缺失值 =====")
    print("缺失值统计:")
    print(df.isnull().sum())
    print(f"已删除 {missing_rows} 行包含缺失值的数据")
    print(f"清洗后剩余 {len(df_dropped)} 行数据")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title(f'{filename} Missing Value Distribution')
    plt.tight_layout()
    plt.show() 
    return df_dropped
def handle_duplicates(df, filename):
    """处理重复行"""
    original_rows = len(df)
    df_clean = df.drop_duplicates()
    duplicate_rows = original_rows - len(df_clean)
    
    print(f"\n===== 处理 {filename} 重复值 =====")
    if duplicate_rows > 0:
        print(f"检测到 {duplicate_rows} 行重复数据，已移除")
    else:
        print("未检测到重复数据")
    print(f"去重后剩余 {len(df_clean)} 行数据")
    return df_clean

def process_data_folder(input_folder, output_folder):
    """处理文件夹中所有CSV文件"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹路径: {output_folder}\n")
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv") and not filename.startswith("cleaned_"):
            file_path = os.path.join(input_folder, filename)
            
            try:

                df = pd.read_csv(file_path)
                print(f"\n===== 开始处理文件: {filename} =====")
                print(f"原始数据行数: {len(df)}")
                df = handle_missing_values(df, filename)
                df = handle_duplicates(df, filename)
                cleaned_filename = f"cleaned_{filename}"
                cleaned_path = os.path.join(output_folder, cleaned_filename)
                df.to_csv(cleaned_path, index=False)
                print(f"\n清洗完成！文件已保存至: {cleaned_path}")
                print("----------------------------------------")
                
            except Exception as e:
                print(f"\n处理 {filename} 时出错: {str(e)}")
                print("----------------------------------------")
                continue
input_folder = r"C:\Users\30444\Desktop\data" 
output_folder = os.path.join(input_folder, "cleaned data") 
print(f"开始处理文件夹: {input_folder} 中的所有CSV文件")
process_data_folder(input_folder, output_folder)
print("\n所有文件处理完成！")
# %%
