# 用于检测模型权重是否与model匹配，仅使用一张图片进行测试，结果保存至test_one_mask文件夹下

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# 导入所有模型
from model.Unet import Unet
from model.Unetpp import UnetPlusPlus
from model.AAU_Net import AAU_Net
from model.utils import *
from model.sgunet import SGUNet
from model.cpfnet import CPFNet
from model.trfe import TRFENet
from model.SmaAtunet.SmaAt_UNet import SmaAt_UNet
from model.attunet import AttU_Net
from model.unext import UNext
from model.deeplab.modeling import deeplabv3plus_resnet101
from model.dscaunet.DCSAU_Net import Model
from model.unext import UNext
from model.ASPPUnet import ASPPUnet
from MCFU_net import MCFU_net


# 参数配置  
parser = argparse.ArgumentParser(description='Single image test for thyroid segmentation')
parser.add_argument('--model_name', type=str, default='MCFU-net', help='模型名称')
parser.add_argument('--dataset', type=str, default='TN3K', choices=['DDTI', 'TN3K'], help='数据集名称')
parser.add_argument('--fold', type=int, default=0, help='交叉验证折数')
parser.add_argument('--epoch', type=int, default=150, help='训练轮次')
parser.add_argument('--best', action='store_true', default=True, help='是否加载best权重')
parser.add_argument('--input_size', type=int, default=256, help='输入尺寸')
parser.add_argument('--img_path', type=str, default=None, help='单张图像路径（可选）')
parser.add_argument('--mask_path', type=str, default=None, help='标签mask路径（可选）')
args = parser.parse_args()

# 数据集路径配置
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATHS = {
    'DDTI': {
        'image': os.path.join(_BASE_DIR, 'data', 'DDTI', '2_preprocessed_data', 'stage1', 'p_image', '1.PNG'),
        'mask':  os.path.join(_BASE_DIR, 'data', 'DDTI', '2_preprocessed_data', 'stage1', 'p_mask', '1.PNG'),
    },
    'TN3K': {
        'image': os.path.join(_BASE_DIR, 'data', 'TN3K', 'test-image', '0000.jpg'),
        'mask':  os.path.join(_BASE_DIR, 'data', 'TN3K', 'test-mask', '0000.jpg'),
    }
}

# 保存结果文件夹
SAVE_DIR = os.path.join(_BASE_DIR, 'test_one_mask')
os.makedirs(SAVE_DIR, exist_ok=True)

# 自动构建权重路径
base_dir = os.path.join(_BASE_DIR, 'weights')
weight_name = f'model_{args.epoch}_best.pth' if args.best else f'model_{args.epoch}.pth'
weight_path = os.path.join(base_dir, args.model_name, args.dataset, f'fold{args.fold}', weight_name)

if not os.path.exists(weight_path):
    raise FileNotFoundError(f'权重文件不存在: {weight_path}')
print(f'加载权重: {weight_path}')

# 根据模型名创建模型
if 'unet' == args.model_name:
    model = Unet(in_ch=3, out_ch=1)
elif 'sgunet' == args.model_name:
    model = SGUNet()
elif 'unext' == args.model_name:
    model = UNext(num_classes=1, img_size=256)
elif 'cpfnet' == args.model_name:
    model = CPFNet()
elif 'trfe' == args.model_name:
    model = TRFENet(3, 1)
elif 'aaunet' == args.model_name:
    model = AAU_Net(in_channels=1, classes_num=2, channels_list=[32, 64, 128, 256, 512])
elif 'ViT' in args.model_name:  # R50-ViT-B_16
    config_vit = CONFIGS_ViT_seg[args.model_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    if args.model_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.input_size / 16), int(args.input_size / 16))
    model = ViT_seg(config_vit, img_size=args.input_size, num_classes=config_vit.n_classes)
elif 'smaatunet' in args.model_name:
    model = SmaAt_UNet(3, 1)
elif 'unext' == args.model_name:
    model = UNext(num_classes=1, img_size=256)
elif 'unetpp' == args.model_name:
    model = UnetPlusPlus(in_ch=3, out_ch=1)
elif 'attunet' == args.model_name:
    model = AttU_Net(in_channel=3, num_classes=1)
elif 'deeplabv3plus' == args.model_name:
    model = deeplabv3plus_resnet101(num_classes=1, output_stride=16)
elif 'dscaunet' == args.model_name:
    model = Model(3, 1)
elif 'deeplabv3plus_8' == args.model_name:
    model = deeplabv3plus_resnet101(num_classes=1, output_stride=8)
elif 'asppunet' == args.model_name:
    model = ASPPUnet(in_ch=3, out_ch=1)
elif 'MCFU-net' == args.model_name:
    model = MCFU_net(in_ch=3, out_ch=1)
    
else:
    raise ValueError(f'未知的模型名： {args.model_name}')

model.load_state_dict(torch.load(weight_path))
model.eval().cuda()

# GPU 信息
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f'使用GPU: {gpu_name}')
else:
    print('警告: 未使用GPU，正在使用CPU')

print(f'模型参数： {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M')

# 自动获取图像和mask路径
if args.img_path is None:
    args.img_path = DATASET_PATHS[args.dataset]['image']
if args.mask_path is None:
    args.mask_path = DATASET_PATHS[args.dataset]['mask']

if not os.path.exists(args.img_path):
    raise FileNotFoundError(f'图像不存在： {args.img_path}')
has_mask = os.path.exists(args.mask_path)
if not has_mask:
    print(f'警告: 标签mask不存在： {args.mask_path}，将只显示原图和预测')

# 读取图像
img = Image.open(args.img_path).convert('RGB')
img = img.resize((256, 256))

if has_mask:
    gt_mask = Image.open(args.mask_path).convert('L')
    gt_mask = gt_mask.resize((256, 256))
    gt_mask_np = np.array(gt_mask).astype(np.float32) / 255.0
    gt_mask_np = (gt_mask_np > 0.5).astype(np.float32)

# 预处理
img_np = np.array(img).astype(np.float32) / 255.0
img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).cuda()

# 推理
with torch.no_grad():
    out = model(img_tensor)

# 后处理
prob = torch.sigmoid(out) if out.min() < 0 else out
prob = prob[0, 0].cpu().numpy()
pred_mask = (prob > 0.5).astype(np.float32)

img_display = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
img_display = np.clip(img_display, 0, 1)

# 可视化
if has_mask:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_display)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(gt_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f'Predicted Mask ({args.model_name} | {args.dataset} | epoch{args.epoch})')
    axes[2].axis('off')
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_display)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f'Predicted Mask ({args.model_name} | {args.dataset} | epoch{args.epoch})')
    axes[1].axis('off')

plt.tight_layout()
save_name = f'{args.model_name}_{args.dataset}_fold{args.fold}_epoch{args.epoch}.png'
save_path = os.path.join(SAVE_DIR, save_name)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f'\n输入形状: {img_tensor.shape}')
print(f'输出形状: {out.shape}')
print(f'可视化已保存: {save_path}')