import numpy as np
import matplotlib.pyplot as plt

# 示例 F1 分数序列（可以替换为实际数据）
f1_scores = [0.0475, 0.2988, 0.4605, 0.5016, 0.5253, 0.5720, 0.5891]

# 计算相邻 epoch 的增长率（N 个 F1 分数会产生 N-1 个增长率）
growth_rates = [f1_scores[i] - f1_scores[i-1] for i in range(1, len(f1_scores))]

# 计算平均增长率
avg_growth = sum(growth_rates) / len(growth_rates)

# 设置 β 参数，例如 0.5
beta = 0.5
threshold = beta * avg_growth

# 取最近一次的增长率作为 g_current
g_current = growth_rates[-1]

# 打印计算结果
print("F1 Scores:", f1_scores)
print("Growth rates:", growth_rates)
print("Average growth rate:", avg_growth)
print("Threshold (beta * average):", threshold)
print("Current growth rate:", g_current)

# 绘图展示
epochs = np.arange(1, len(f1_scores))  # 表示增长率对应的 epoch 差值（从 epoch1 到 epochN）
plt.figure(figsize=(10, 6))
plt.plot(epochs, growth_rates, marker='o', linestyle='-', color='blue', label="Growth rate")
plt.axhline(y=avg_growth, color='green', linestyle='--', label=f"Avg growth ({avg_growth:.4f})")
plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold (beta*avg = {threshold:.4f})")
plt.scatter(epochs[-1], g_current, color='orange', s=100, label=f"Current growth ({g_current:.4f})")
plt.xlabel("Epoch Index (for growth rate)")
plt.ylabel("Growth rate")
plt.title("F1 Score Growth Rates and Decision Threshold")
plt.legend()
plt.grid(True)
plt.show()

# 判断是否触发数据集切换
if g_current < threshold:
    print("最新增长率低于阈值，触发数据集切换。")
else:
    print("最新增长率未低于阈值，保持当前数据集。")
