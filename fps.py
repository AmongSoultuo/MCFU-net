import torch
import numpy as np
from torchvision import models
from thop import profile
import torch
from model.Unet import Unet
from model.Unetpp import UnetPlusPlus
from model.AAU_Net import AAU_Net
from model.our_model import our_net
from model.utils import *
from model.sgunet import SGUNet
from model.cpfnet import CPFNet
from model.trfe import TRFENet
from model.attunet import AttU_Net
from model.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.SmaAtunet.SmaAt_UNet import SmaAt_UNet
from model.deeplab_50.modeling import deeplabv3plus_resnet50
from model.dscaunet.DCSAU_Net import Model
from model.unext import UNext
model_name = 'R50-ViT-B_16'
config_vit = CONFIGS_ViT_seg[model_name]
config_vit.n_classes = 1
config_vit.n_skip = 3  # 这里的n_skip含义不明,R50为3,别的用0
if model_name.find('R50') != -1:
    config_vit.patches.grid = (
        int(256 / 16), int(256 / 16))
model = Unet(3, 1)
device = torch.device("cuda")  # 将模型移动到 CUDA 设备
model.to(device)
dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float).to(device)
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

# GPU 预热
for _ in range(10):
    _ = model(dummy_input)

# 测量性能
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()  # 等待 GPU 同步
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000.0 / mean_syn
print('Mean inference time: {:.3f}ms'.format(mean_syn))
print('Standard deviation: {:.3f}ms'.format(std_syn))
print('Frames per second: {:.2f}'.format(mean_fps))
