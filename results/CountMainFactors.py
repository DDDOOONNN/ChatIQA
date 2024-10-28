import pandas as pd
import re
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'image_assessments.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 定义要提取的关键词模式
pattern = r'\*\*(.*?)\:\*\*'

# 初始化计数器
factor_counts = {}

# 统计每个因素的个数
for index, row in df.iterrows():
    text = row.iloc[1]  # 使用 iloc 获取第二列的文本
    matches = re.findall(pattern, text)
    for match in matches:
        factor = match.strip()  # 去除多余的空格
        if factor in factor_counts:
            factor_counts[factor] += 1
        else:
            factor_counts[factor] = 1

# 将计数转换为DataFrame并排序
sorted_factors = pd.DataFrame(factor_counts.items(), columns=['Factor', 'Count'])
sorted_factors = sorted_factors.sort_values(by='Count', ascending=False).head(30)

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(sorted_factors['Factor'], sorted_factors['Count'], color='skyblue')
plt.xlabel('Occurrence Count')
plt.title('Top 30 Image Quality Factors by Occurrence')
plt.gca().invert_yaxis()  # 反转Y轴以显示最高的在上面
plt.show()
