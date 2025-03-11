import pandas as pd
from datetime import datetime, timedelta

# 读取 CSV 文件（请将 "data.csv" 替换为你的文件名）
df = pd.read_csv("Final_data/ABE2019_utide(20,23,25,27)(1final).csv")

# 只保留 future_steps 为 25 的行
df_25 = df[df["future_steps"] == 25].copy()

# 定义起始时间为 01-01 00:00，注意这里只需要月日和时间，所以年份可以随意设置
start_time = datetime.strptime("01-01 00:00", "%m-%d %H:%M")

# 将 Time_Step 列转换为以起始时间为基础、每步递增 15 分钟的格式
df_25["Time_Step"] = df_25["Time_Step"].apply(lambda x: (start_time + timedelta(minutes=15 * int(x))).strftime("%m-%d %H:%M"))

# 将结果保存为新的 CSV 文件
df_25.to_csv("ABE_UB.csv", index=False)

print("处理完成，已生成 'filtered_data.csv' 文件")