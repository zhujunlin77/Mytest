"""
DSEC数据集加载器
用于读取DSEC数据集的事件流数据和RGB图像
"""

import os
import h5py
import hdf5plugin  # 用于支持DSEC数据集的压缩格式
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, Tuple, List
import glob


class DSECLoader:
    """
    DSEC数据集加载器
    
    用于加载和处理DSEC数据集的事件流数据和RGB图像
    """
    
    def __init__(self, data_path: str):
        """
        初始化DSEC加载器
        
        Args:
            data_path: DSEC数据集路径
        """
        self.data_path = Path(data_path)
        
        # 事件流数据路径
        self.events_dir = self.data_path / "interlaken_00_c_events_left"
        self.events_file = self.events_dir / "events.h5"
        
        # RGB图像路径
        self.images_dir = self.data_path / "interlaken_00_c_images_rectified_left"
        
        # 检查路径是否存在
        if not self.events_file.exists():
            raise FileNotFoundError(f"事件文件未找到: {self.events_file}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图像目录未找到: {self.images_dir}")
        
        # 加载事件数据
        self.h5f = None
        self.events_data = None
        self.load_events()
        
        # 加载图像列表
        self.image_files = self._load_image_list()
        
        print(f"成功加载DSEC数据集")
        print(f"事件文件: {self.events_file}")
        print(f"找到 {len(self.image_files)} 张RGB图像")
        
    def load_events(self):
        """加载事件流数据 (延迟加载模式)"""
        try:
            self.h5f = h5py.File(str(self.events_file), 'r')
            
            # 保持h5数据集引用，不立即加载到内存
            self.events_data = {
                'x': self.h5f['events/x'],  # x坐标
                'y': self.h5f['events/y'],  # y坐标
                'p': self.h5f['events/p'],  # 极性 (0/1)
                't': self.h5f['events/t']   # 时间戳 (微秒)
            }
            
            # 获取基本统计信息（只读取少量数据）
            total_events = len(self.events_data['t'])
            
            # 读取前1000个事件用于统计
            sample_size = min(1000, total_events)
            t_first = self.events_data['t'][0]
            t_last = self.events_data['t'][total_events - 1]
            x_sample = self.events_data['x'][:sample_size]
            y_sample = self.events_data['y'][:sample_size]
            p_sample = self.events_data['p'][:sample_size]
            
            print(f"事件数据统计:")
            print(f"  总事件数: {total_events:,}")
            print(f"  时间范围: {t_first} - {t_last} (微秒)")
            print(f"  持续时间: {(t_last - t_first) / 1e6:.2f} 秒")
            print(f"  x范围 (采样): {x_sample.min()} - {x_sample.max()}")
            print(f"  y范围 (采样): {y_sample.min()} - {y_sample.max()}")
            print(f"  极性分布 (采样): 正={np.sum(p_sample == 1)}, 负={np.sum(p_sample == 0)}")
            
        except Exception as e:
            print(f"加载事件数据时出错: {e}")
            raise
    
    def _load_image_list(self) -> List[Path]:
        """加载图像文件列表"""
        image_files = sorted(list(self.images_dir.glob("*.png")))
        return image_files
    
    def get_events(self, start_idx: int = 0, end_idx: int = None) -> Dict[str, np.ndarray]:
        """
        获取指定范围的事件数据
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引 (None表示到末尾)
            
        Returns:
            包含x, y, p, t的字典
        """
        if end_idx is None:
            end_idx = len(self.events_data['t'])
        
        return {
            'x': self.events_data['x'][start_idx:end_idx],
            'y': self.events_data['y'][start_idx:end_idx],
            'p': self.events_data['p'][start_idx:end_idx],
            't': self.events_data['t'][start_idx:end_idx]
        }
    
    def get_events_by_time(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """
        根据时间戳范围获取事件数据
        
        Args:
            t_start_us: 起始时间 (微秒)
            t_end_us: 结束时间 (微秒)
            
        Returns:
            包含x, y, p, t的字典
        """
        # 读取时间数组并找到索引范围
        t_array = np.array(self.events_data['t'])
        
        # 找到时间范围内的索引
        start_idx = np.searchsorted(t_array, t_start_us, side='left')
        end_idx = np.searchsorted(t_array, t_end_us, side='right')
        
        return {
            'x': np.array(self.events_data['x'][start_idx:end_idx]),
            'y': np.array(self.events_data['y'][start_idx:end_idx]),
            'p': np.array(self.events_data['p'][start_idx:end_idx]),
            't': np.array(self.events_data['t'][start_idx:end_idx])
        }
    
    def get_image(self, index: int) -> np.ndarray:
        """
        加载指定索引的RGB图像
        
        Args:
            index: 图像索引
            
        Returns:
            RGB图像 (numpy数组)
        """
        if index < 0 or index >= len(self.image_files):
            raise IndexError(f"图像索引超出范围: {index} (总共 {len(self.image_files)} 张)")
        
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        
        return image
    
    def get_image_by_name(self, filename: str) -> np.ndarray:
        """
        根据文件名加载RGB图像
        
        Args:
            filename: 图像文件名 (例如 "000000.png")
            
        Returns:
            RGB图像 (numpy数组)
        """
        image_path = self.images_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件未找到: {image_path}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        
        return image
    
    def visualize_events(self, events: Dict[str, np.ndarray], 
                        height: int = 480, width: int = 640,
                        method: str = 'color') -> np.ndarray:
        """
        将事件数据可视化为图像
        
        Args:
            events: 事件数据字典
            height: 图像高度
            width: 图像宽度
            method: 可视化方法 ('color' 或 'grayscale')
            
        Returns:
            可视化图像 (numpy数组)
        """
        if method == 'color':
            # 分别统计正负极性事件
            pos_image = np.zeros((height, width), dtype=np.float32)
            neg_image = np.zeros((height, width), dtype=np.float32)
            
            # 累加事件
            for i in range(len(events['x'])):
                x, y, p = int(events['x'][i]), int(events['y'][i]), events['p'][i]
                if 0 <= x < width and 0 <= y < height:
                    if p == 1:  # 正极性
                        pos_image[y, x] += 1
                    else:  # 负极性
                        neg_image[y, x] += 1
            
            # 应用对数缩放以增强对比度
            pos_image = np.log1p(pos_image)
            neg_image = np.log1p(neg_image)
            
            # 归一化到0-1范围
            if pos_image.max() > 0:
                pos_image = pos_image / pos_image.max()
            if neg_image.max() > 0:
                neg_image = neg_image / neg_image.max()
            
            # 创建RGB图像：正极性=红色，负极性=蓝色
            event_image = np.zeros((height, width, 3), dtype=np.float32)
            event_image[:, :, 0] = pos_image  # 红色通道
            event_image[:, :, 2] = neg_image  # 蓝色通道
            
            # 应用gamma校正以增强可视化效果
            gamma = 0.5  # 小于1会增强暗部细节
            event_image = np.power(event_image, gamma)
            
            # 转换到0-255范围
            event_image = (event_image * 255).astype(np.uint8)
            
        else:  # grayscale
            # 灰度图可视化
            event_image = np.zeros((height, width), dtype=np.float32)
            
            for i in range(len(events['x'])):
                x, y = int(events['x'][i]), int(events['y'][i])
                if 0 <= x < width and 0 <= y < height:
                    event_image[y, x] += 1
            
            # 对数缩放
            event_image = np.log1p(event_image)
            
            # 归一化
            if event_image.max() > 0:
                event_image = event_image / event_image.max() * 255
            
            event_image = event_image.astype(np.uint8)
        
        return event_image
    
    def get_num_images(self) -> int:
        """返回图像总数"""
        return len(self.image_files)
    
    def get_num_events(self) -> int:
        """返回事件总数"""
        return len(self.events_data['t'])
    
    def get_time_range(self) -> Tuple[int, int]:
        """返回事件时间范围 (微秒)"""
        return self.events_data['t'][0], self.events_data['t'][-1]
    
    def __del__(self):
        """析构函数，关闭h5文件"""
        if self.h5f is not None:
            self.h5f.close()
    
    def close(self):
        """手动关闭h5文件"""
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None


def demo():
    """演示如何使用DSECLoader"""
    
    # 数据集路径
    data_path = r"c:\Users\zjl\Desktop\events_simulaters\WGSE-SpikeCamera-main\DSEC-data"
    
    # 创建加载器
    loader = DSECLoader(data_path)
    
    print("\n=== 数据集信息 ===")
    print(f"图像总数: {loader.get_num_images()}")
    print(f"事件总数: {loader.get_num_events():,}")
    t_start, t_end = loader.get_time_range()
    print(f"时间范围: {t_start} - {t_end} 微秒 (约 {(t_end - t_start) / 1e6:.2f} 秒)")
    
    # 读取第一张图像
    print("\n=== 读取第一张RGB图像 ===")
    image_0 = loader.get_image(0)
    print(f"图像形状: {image_0.shape}")
    print(f"图像数据类型: {image_0.dtype}")
    
    # 读取前1000个事件
    print("\n=== 读取前1000个事件 ===")
    events_sample = loader.get_events(0, 1000)
    print(f"x: {events_sample['x'][:10]}...")
    print(f"y: {events_sample['y'][:10]}...")
    print(f"p: {events_sample['p'][:10]}...")
    print(f"t: {events_sample['t'][:10]}...")
    
    # 根据时间窗口读取事件
    print("\n=== 读取时间窗口内的事件 ===")
    t_window_start = t_start
    t_window_end = t_start + 100000  # 100ms窗口
    events_window = loader.get_events_by_time(t_window_start, t_window_end)
    print(f"时间窗口 [{t_window_start}, {t_window_end}] 内的事件数: {len(events_window['t'])}")
    
    # 可视化事件
    print("\n=== 可视化事件 ===")
    event_image = loader.visualize_events(events_window)
    print(f"事件图像形状: {event_image.shape}")
    
    # 关闭加载器
    loader.close()
    print("\n完成!")


if __name__ == "__main__":
    demo()
