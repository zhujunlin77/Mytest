import os

import numpy as np
import cv2
import torch
import argparse

from dwtnets import Dwt1dResnetX_TCN
from utils import RawToSpike
from einops import rearrange
from pytorch_wavelets import DWT1DForward


parser = argparse.ArgumentParser(description='AAAI - WGSE - REDS')
parser.add_argument('-w', '--wvl', type=str, default='db8', help='select wavelet base function')
parser.add_argument('-j', '--jlevels', type=int, default=5)
parser.add_argument('-k', '--kernel_size', type=int, default=3)
parser.add_argument('-l', '--logpath', type=str, default='WGSE-Dwt1dNet')
parser.add_argument('-f', '--datfile', type=str, default=None, help='path of the spike data to be tested')

args = parser.parse_args()
dataroot = args.datfile
wvlname = args.wvl
j = args.jlevels
logfolder = args.logpath
ks = args.kernel_size


def nor(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def progress_bar_time(total_time):
    hour = int(total_time) // 3600
    minu = (int(total_time) % 3600) // 60
    sec = int(total_time) % 60
    return '%d:%02d:%02d' % (hour, minu, sec)


def main():

    f = open(dataroot, 'rb')
    spike_seq = f.read()
    spike_seq = np.frombuffer(spike_seq, 'b')
    spikes = RawToSpike(spike_seq, 250, 400)
    spikes = spikes.astype(np.float32)
    spikes = torch.from_numpy(spikes)
    f.close()

    spikes = spikes[None, :, :, :]
    total_t = spikes.shape[1]
    
    if total_t < 41:
        print(f"Warning: Input spike data has only {total_t} time steps, which is less than required 41. Skipping.")
        return
    
    # Use middle 41 time steps
    start_idx = (total_t - 41) // 2
    end_idx = start_idx + 41
    spikes = spikes[:, start_idx:end_idx, :, :]
    
    s = spikes[:, :, 0:1, 0:1]
    dwt = DWT1DForward(wave=wvlname, J=j)
    s_r = rearrange(s, 'b t h w -> b h w t')
    s_r = rearrange(s_r, 'b h w t -> (b h w) 1 t')
    yl, yh = dwt(s_r)
    yl_size = yl.shape[-1]
    yh_size = [yhi.shape[-1] for yhi in yh]
    
    model = Dwt1dResnetX_TCN(
        wvlname=wvlname, J=j, yl_size=yl_size, yh_size=yh_size, num_residual_blocks=3, norm=None, ks=ks, store_features=True
    )
    print(model)

    saved_state_dict = torch.load(logfolder)
    model.load_state_dict(saved_state_dict.module.state_dict())
    
    model = model.cuda()
    model.eval()

    pred = model(spikes.cuda())        
    prediction = pred[0].permute(1,2,0).cpu().detach().numpy()
    
    # Save demo.png in the same directory as the model file
    model_dir = os.path.dirname(logfolder)
    output_path = os.path.join(model_dir, 'demo.png')
    cv2.imwrite(output_path, prediction * 255.0)
    print(f"Demo image saved to: {output_path}")
    

if __name__ == '__main__':
    main()

    