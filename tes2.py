import collections

# 读取文件
with open('simple-error-model.tr', 'r') as f:
    lines = f.readlines()

# 分析丢包
drop_count = 0
drop_details = collections.defaultdict(int)
drop_times = []

for line in lines:
    if line.startswith('d'):
        parts = line.split()
        if len(parts) >= 3:
            drop_count += 1
            time = float(parts[1])
            drop_times.append(time)
            node_path = parts[2]
            drop_details[node_path] += 1

# 输出结果
print(f"总计丢包数量: {drop_count}")
print("\n按节点统计:")
for node, count in drop_details.items():
    print(f"{node}: {count}次丢包")

# 绘制丢包时间分布 (需matplotlib)
if drop_times:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.hist(drop_times, bins=50, alpha=0.7)
    plt.title('丢包时间分布')
    plt.xlabel('时间 (秒)')
    plt.ylabel('丢包次数')
    plt.grid(True)
    plt.savefig('drop_time_distribution.png')