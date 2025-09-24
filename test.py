import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置支持中文的字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Packet:
    """存储单个数据包的完整生命周期信息"""
    def __init__(self, src_ip, dst_ip, pkt_id):
        self.src_ip = src_ip        # 源IP
        self.dst_ip = dst_ip        # 目的IP
        self.pkt_id = pkt_id        # 包ID（用于唯一标识）
        self.events = defaultdict(list)  # 事件类型: [时间戳列表]
        self.paths = []             # 路径记录（事件类型+路径）

    def add_event(self, event_type, timestamp, path):
        """添加事件记录"""
        self.events[event_type].append(timestamp)
        self.paths.append((event_type, path))

    @property
    def e2e_delay(self):
        """计算端到端延迟（从入队到接收）"""
        if 'enqueue' in self.events and 'receive' in self.events:
            # 取第一个入队和最后一个接收事件的时间差
            enqueue_time = self.events['enqueue'][0]
            receive_time = self.events['receive'][-1]
            return receive_time - enqueue_time
        return None

    @property
    def queue_delay(self):
        """计算队列处理延迟（从入队到出队）"""
        if 'enqueue' in self.events and 'dequeue' in self.events:
            # 取第一个入队和第一个出队事件的时间差
            enqueue_time = self.events['enqueue'][0]
            dequeue_time = self.events['dequeue'][0]
            return dequeue_time - enqueue_time
        return None

    @property
    def transmission_delay(self):
        """计算传输延迟（从出队到接收）"""
        if 'dequeue' in self.events and 'receive' in self.events:
            dequeue_time = self.events['dequeue'][0]
            receive_time = self.events['receive'][-1]
            return receive_time - dequeue_time
        return None

# 全局统计变量
packet_stats = defaultdict(Packet)  # 按（源IP,目的IP,包ID）存储数据包
sent_packets = 0                  # 总发送包数
dropped_packets = 0               # 总丢失包数（无接收事件）
received_packets = 0              # 总接收包数

def parse_tr_file(filename):
    global sent_packets,dropped_packets,received_packets
    """解析TR文件，提取数据包事件信息"""
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # 分割行内容（事件类型、时间戳、路径、包信息）
                parts = line.split(maxsplit=3)
                if len(parts) < 4:
                    print(f"警告：行 {line_num} 格式不完整，跳过")
                    continue

                event_type, timestamp_str, path, packet_info = parts
                timestamp = float(timestamp_str)  # 转换时间戳为浮点数

                # 提取IP头和包ID（关键正则表达式）
                ip_match = re.search(
                    r'ns3::Ipv4Header.*?(\d+\.\d+\.\d+\.\d+)\s*>\s*(\d+\.\d+\.\d+\.\d+).*?\sid\s(\d+)',
                    packet_info
                )
                if not ip_match:
                    print(f"警告：行 {line_num} 无法解析IP头，跳过")
                    continue

                src_ip, dst_ip, pkt_id = ip_match.groups()
                packet_key = (src_ip, dst_ip, pkt_id)

                # 更新全局计数器
                if event_type == '+':
                    sent_packets += 1
                elif event_type == 'd':
                    dropped_packets += 1
                elif event_type == 'r':
                    received_packets += 1

                # 映射事件类型别名（方便后续处理）
                event_alias = {
                    '+': 'enqueue',   # 入队事件
                    '-': 'dequeue',   # 出队事件
                    'r': 'receive'    # 接收事件
                }.get(event_type, event_type)

                # 添加事件记录到数据包对象
                if packet_key not in packet_stats:
                    packet_stats[packet_key] = Packet(src_ip, dst_ip, pkt_id)
                packet_stats[packet_key].add_event(event_alias, timestamp, path)

            except Exception as e:
                print(f"错误：解析行 {line_num} 时发生异常 - {str(e)}")
                continue

def calculate_metrics():
    """计算关键性能指标"""
    e2e_delays = []
    queue_delays = []
    transmission_delays = []
    valid_packets = 0  # 有效数据包数（有接收事件的）

    for pkt in packet_stats.values():
        # 端到端延迟
        e2e_delay = pkt.e2e_delay
        if e2e_delay is not None:
            e2e_delays.append(e2e_delay)
            valid_packets += 1

        # 队列延迟
        queue_delay = pkt.queue_delay
        if queue_delay is not None:
            queue_delays.append(queue_delay)

        # 传输延迟
        transmission_delay = pkt.transmission_delay
        if transmission_delay is not None:
            transmission_delays.append(transmission_delay)

    # 计算丢包率（丢失包数/总发送包数）
    loss_rate = dropped_packets / sent_packets if sent_packets > 0 else 0

    return {
        'total_sent': sent_packets,
        'total_received': received_packets,
        'total_dropped': dropped_packets,
        'loss_rate': loss_rate,
        'e2e_delays': e2e_delays,
        'queue_delays': queue_delays,
        'transmission_delays': transmission_delays,
        'valid_packets': valid_packets
    }

def visualize_results(metrics):
    """可视化分析结果"""
    plt.figure(figsize=(18, 12))

    # 1. 端到端延迟分布
    plt.subplot(2, 2, 1)
    if metrics['e2e_delays']:
        plt.hist(metrics['e2e_delays'], bins=50, alpha=0.7, color='skyblue')
        plt.title('端到端延迟分布 (秒)')
        plt.xlabel('延迟 (秒)')
        plt.ylabel('包数量')
    else:
        plt.text(0.5, 0.5, "无有效端到端延迟数据", ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 2. 队列延迟分布
    plt.subplot(2, 2, 2)
    if metrics['queue_delays']:
        plt.hist(metrics['queue_delays'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('队列处理延迟分布 (秒)')
        plt.xlabel('延迟 (秒)')
        plt.ylabel('包数量')
    else:
        plt.text(0.5, 0.5, "无有效队列延迟数据", ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 3. 传输延迟分布
    plt.subplot(2, 2, 3)
    if metrics['transmission_delays']:
        plt.hist(metrics['transmission_delays'], bins=50, alpha=0.7, color='salmon')
        plt.title('传输延迟分布 (秒)')
        plt.xlabel('延迟 (秒)')
        plt.ylabel('包数量')
    else:
        plt.text(0.5, 0.5, "无有效传输延迟数据", ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 4. 丢包统计
    plt.subplot(2, 2, 4)
    labels = ['已发送', '已接收', '已丢失']
    values = [metrics['total_sent'], metrics['total_received'], metrics['total_dropped']]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    plt.bar(labels, values, color=colors)
    plt.title('数据包状态统计')
    plt.ylabel('包数量')
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center')

    plt.tight_layout()
    plt.savefig('network_performance.png', dpi=300)
    plt.show()

def main():
    # 配置参数
    tr_file = "simple-error-model.tr"  # 替换为你的TR文件路径

    # 1. 解析TR文件
    print(f"开始解析TR文件: {tr_file}...")
    parse_tr_file(tr_file)

    # 2. 计算性能指标
    print("\n计算性能指标...")
    metrics = calculate_metrics()

    # 3. 输出统计结果
    print("\n===== 性能分析报告 =====")
    print(f"总发送包数: {metrics['total_sent']}")
    print(f"总接收包数: {metrics['total_received']}")
    print(f"总丢失包数: {metrics['total_dropped']}")
    print(f"丢包率: {metrics['loss_rate'] * 100:.2f}%")
    print(f"有效数据包数（有接收事件）: {metrics['valid_packets']}")

    if metrics['e2e_delays']:
        print(f"\n端到端延迟统计:")
        print(f"  平均延迟: {np.mean(metrics['e2e_delays']):.6f} 秒")
        print(f"  最大延迟: {max(metrics['e2e_delays']):.6f} 秒")
        print(f"  最小延迟: {min(metrics['e2e_delays']):.6f} 秒")
    else:
        print("\n无有效端到端延迟数据")

    # 4. 可视化结果
    print("\n生成可视化报告...")
    visualize_results(metrics)

if __name__ == "__main__":
    main()