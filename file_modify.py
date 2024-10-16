import os
import pandas as pd

# 输入文件夹路径，放置 .txt 文件
input_folder = 'path_to_txt_files'
# 输出文件夹路径，存放 .csv 文件
output_folder = 'path_to_csv_files'

# 遍历输入文件夹下的所有 .txt 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # 构建完整的 .txt 文件路径
        txt_file_path = os.path.join(input_folder, filename)
        
        # 使用 pandas 读取 .txt 文件并删除前 11 行
        df = pd.read_csv(txt_file_path, sep=' ', header=None, skiprows=11)  # 根据分隔符调整 sep 参数
        
        # 构建输出的 .csv 文件路径
        csv_file_path = os.path.join(output_folder, filename.replace('.txt', '.csv'))
        
        # 将 DataFrame 保存为 .csv 文件
        df.to_csv(csv_file_path, index=False)
        
        print(f"Converted {filename} to CSV and saved to {csv_file_path}")

print("All files have been processed.")
