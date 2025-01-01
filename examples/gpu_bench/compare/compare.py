import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 加载 CSV 文件
k100_file_path = 'latency_data_k100.csv'
n3080_file_path = 'latency_data_3080.csv'

# 读取 CSV 文件
k100_data = pd.read_csv(k100_file_path)
n3080_data = pd.read_csv(n3080_file_path)

print(k100_data)

# 假设CSV文件具有 'prompt_len', 'prefill_latency', 'decode_latency' 列
# 提取数据
prompt_lengths_k100 = k100_data['prompt_len']
prefill_latencies_k100 = k100_data['prefill_latency']
decode_latencies_k100 = k100_data['decode_latency']

prompt_lengths_n3080 = n3080_data['prompt_len']
prefill_latencies_n3080 = n3080_data['prefill_latency']
decode_latencies_n3080 = n3080_data['decode_latency']

# 创建绘图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# K100的延迟图
axs[0].plot(prompt_lengths_k100, prefill_latencies_k100, label='K100', marker='o', linestyle='-', color='b')
axs[0].plot(prompt_lengths_n3080, prefill_latencies_n3080, label='NV 3080', marker='x', linestyle='-', color='r')
axs[0].set_title('Prefill Latency')
axs[0].set_xlabel('Prompt Length')
axs[0].set_ylabel('Latency (seconds)')
axs[0].legend(loc='upper left')
axs[0].grid(True)

# n3080的延迟图
axs[1].plot(prompt_lengths_k100, decode_latencies_k100, label='K100', marker='o', linestyle='-', color='b')
axs[1].plot(prompt_lengths_n3080, decode_latencies_n3080, label='NV 3080', marker='x', linestyle='-', color='r')
axs[1].set_title('Decode Latency')
axs[1].set_xlabel('Prompt Length')
axs[1].set_ylabel('Latency (seconds)')
axs[1].legend(loc='upper left')
axs[1].grid(True)

# 调整布局
plt.tight_layout()

# 显示图表
plt.savefig('latency_vs_prompt_len.png')