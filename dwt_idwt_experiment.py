import numpy as np
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from einops import rearrange
import utils
import os

def load_spike_data(dat_path, H=250, W=400):
    """加载脉冲数据文件"""
    print(f"加载脉冲数据: {dat_path}")
    f = open(dat_path, 'rb')
    spike_seq = f.read()
    spike_seq = np.frombuffer(spike_seq, 'b')
    spikes = utils.RawToSpike(spike_seq, H, W)
    f.close()
    
    # 转换为float32并归一化
    spikes = spikes.astype(np.float32)
    print(f"原始数据形状: {spikes.shape}")  # (T, H, W)
    return spikes

def dwt_idwt_experiment():
    """DWT和IDWT实验"""
    # 参数设置
    dat_path = 'test_dataset/train/input/000_part1_key_id21.dat'
    wavelet = 'db8'
    J = 8  # 小波变换层数
    H, W = 250, 400  # 数据的空间维度
    
    # 检查文件是否存在
    if not os.path.exists(dat_path):
        print(f"错误: 文件 {dat_path} 不存在")
        return
    
    # 加载数据
    spikes = load_spike_data(dat_path, H, W)
    
    # 转换为PyTorch张量并调整维度: (T, H, W) -> (1, T, H, W)
    spikes_tensor = torch.from_numpy(spikes).unsqueeze(0)
    print(f"输入张量形状: {spikes_tensor.shape}")
    
    # 创建DWT和IDWT模块
    dwt = DWT1DForward(J=J, wave=wavelet)
    idwt = DWT1DInverse(wave=wavelet)
    
    # 仿照dwtnets.py中的forward方法进行维度变换
    B, T, H, W = spikes_tensor.shape
    print(f"批次大小: {B}, 时间维度: {T}, 高度: {H}, 宽度: {W}")
    
    # 将时间维度移到最后: (B, T, H, W) -> (B, H, W, T)
    x_r = rearrange(spikes_tensor, 'b t h w -> b h w t')
    print(f"维度变换1后: {x_r.shape}")
    
    # 把空间维度合并到批次维度: (B, H, W, T) -> (B*H*W, 1, T)
    x_r = rearrange(x_r, 'b h w t -> (b h w) 1 t')
    print(f"维度变换2后: {x_r.shape}")
    
    # 执行DWT变换
    print("\n执行DWT变换...")
    yl, yh = dwt(x_r)
    print(f"低通系数形状: {yl.shape}")
    print(f"高通系数层数: {len(yh)}")
    for i, yhi in enumerate(yh):
        print(f"  第{i+1}层高通系数形状: {yhi.shape}")
    
    # 执行IDWT逆变换（直接使用DWT得到的系数）
    print("\n执行IDWT逆变换...")
    reconstructed = idwt((yl, yh))
    print(f"重建数据形状: {reconstructed.shape}")
    
    # 维度变换回原始形状
    # (B*H*W, 1, T) -> (B, H, W, T)
    out = rearrange(reconstructed, '(b h w) 1 t -> b h w t', b=B, h=H, w=W)
    print(f"维度变换3后: {out.shape}")
    
    # (B, H, W, T) -> (B, T, H, W)
    out = rearrange(out, 'b h w t -> b t h w')
    print(f"最终输出形状: {out.shape}")
    
    # 移除批次维度并转换为numpy
    reconstructed_spikes = out.squeeze(0).numpy()
    print(f"重建数据形状: {reconstructed_spikes.shape}")
    
    # 计算重建误差
    original_spikes = spikes_tensor.squeeze(0).numpy()
    
    # 处理形状不匹配的情况（DWT/IDWT可能导致长度变化）
    if original_spikes.shape[0] != reconstructed_spikes.shape[0]:
        print(f"警告: 原始数据时间维度({original_spikes.shape[0]})与重建数据({reconstructed_spikes.shape[0]})不匹配")
        
        # 裁剪较长的数组使其匹配
        min_time = min(original_spikes.shape[0], reconstructed_spikes.shape[0])
        original_spikes = original_spikes[:min_time, :, :]
        reconstructed_spikes = reconstructed_spikes[:min_time, :, :]
        print(f"裁剪后形状: {original_spikes.shape}")
    
    mse = np.mean((original_spikes - reconstructed_spikes) ** 2)
    print(f"\n重建MSE: {mse:.6f}")
    print(f"重建RMSE: {np.sqrt(mse):.6f}")
    
    # 检查数据范围
    print(f"\n原始数据范围: [{original_spikes.min():.4f}, {original_spikes.max():.4f}]")
    print(f"重建数据范围: [{reconstructed_spikes.min():.4f}, {reconstructed_spikes.max():.4f}]")
    
    # # 保存结果
    # output_dir = 'logs/dwt_idwt_experiment'
    # os.makedirs(output_dir, exist_ok=True)
    
    # # 保存原始和重建数据
    # np.save(os.path.join(output_dir, 'original_spikes.npy'), original_spikes)
    # np.save(os.path.join(output_dir, 'reconstructed_spikes.npy'), reconstructed_spikes)
    
    # # 保存小波系数
    # np.save(os.path.join(output_dir, 'yl_coeff.npy'), yl.numpy())
    # for i, yhi in enumerate(yh):
    #     np.save(os.path.join(output_dir, f'yh_coeff_layer{i+1}.npy'), yhi.numpy())
    
    # print(f"\n实验结果已保存到: {output_dir}")
    # print("文件包括:")
    # print("  - original_spikes.npy: 原始脉冲数据")
    # print("  - reconstructed_spikes.npy: 重建的脉冲数据")
    # print("  - yl_coeff.npy: 低通小波系数")
    # print("  - yh_coeff_layerX.npy: 各层高通小波系数")
    
    return original_spikes, reconstructed_spikes, yl, yh

if __name__ == "__main__":
    print("=" * 60)
    print("DWT/IDWT 实验")
    print("=" * 60)
    print(f"小波函数: db8")
    print(f"变换层数: 8")
    print(f"输入数据: test_dataset/train/input/000_part1_key_id21.dat")
    print("=" * 60)
    
    dwt_idwt_experiment()
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)