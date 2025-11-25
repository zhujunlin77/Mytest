"""
DSEC数据集可视化演示
演示如何加载并可视化DSEC数据集的事件流和RGB图像
"""

import numpy as np
import matplotlib.pyplot as plt
from load_dsec import DSECLoader

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def visualize_dsec_data():
    """可视化DSEC数据集的事件流和RGB图像"""
    
    # 数据集路径
    data_path = r"c:\Users\zjl\Desktop\events_simulaters\WGSE-SpikeCamera-main\DSEC-data"
    
    # 创建加载器
    loader = DSECLoader(data_path)
    
    # 获取时间范围
    t_start, t_end = loader.get_time_range()
    print(f"\n数据集时间范围: {t_start} - {t_end} 微秒")
    print(f"总时长: {(t_end - t_start) / 1e6:.2f} 秒")

    
    # 设置可视化参数
    num_samples = 5  # 显示5个时间窗口
    window_duration_ms = 50  # 每个窗口50ms
    
    # 创建图形
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('DSEC数据集可视化: 事件流 vs RGB图像', fontsize=14)
    
    # 在不同时间点采样
    for i in range(num_samples):
        # 计算时间窗口
        t_window_start = window_duration_ms * 1000 * i
        t_window_end = t_window_start + window_duration_ms * 1000
        
        print(f"\n处理时间窗口 {i+1}/{num_samples}")
        print(f"  时间范围: [{t_window_start}, {t_window_end}] (微秒)")
        
        # 获取事件数据
        events = loader.get_events_by_time(t_window_start, t_window_end)
        print(f"  事件数: {len(events['t']):,}")
        
        # 可视化事件
        event_image = loader.visualize_events(events, height=480, width=640)
        axes[0, i].imshow(event_image)
        axes[0, i].set_title(f'事件 @ {t_window_start/1000:.1f}ms\n({len(events["t"]):,} events)')
        axes[0, i].axis('off')
        
        # 获取对应的RGB图像（按时间比例选择）
        image_idx = i
        rgb_image = loader.get_image(image_idx)
        
        # 调整RGB图像大小以匹配事件图像
        import cv2
        rgb_resized = cv2.resize(rgb_image, (640, 480))
        
        axes[1, i].imshow(rgb_resized)
        axes[1, i].set_title(f'RGB 图像 #{image_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'dsec_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_path}")
    
    # plt.show()  # 注释掉以避免阻塞
    plt.close()  # 关闭图形以释放内存
    
    # 关闭加载器
    loader.close()
    print("\n完成!")



if __name__ == "__main__":
    print("=" * 60)
    print("DSEC数据集可视化演示")
    print("=" * 60)
    
    # 运行可视化
    visualize_dsec_data()
