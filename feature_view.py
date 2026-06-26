import argparse
import copy
import os
import time
from torchstat import stat
import matplotlib.pyplot as plt
import numpy as np
import cv2
# PyTorch includes
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from metrics import plot_curves
# Dataloaders includes
from dataloaders import custom_transforms as trforms
from dataloaders.ddti import DDTI
from dataloaders.tn3k import TN3K
from torchvision import transforms
# Model includes
from model.Unet import Unet
from model.Unetpp import UnetPlusPlus
from model.AAU_Net import AAU_Net
from model.our_model import our_net
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
from metrics import evaluate
import numpy as np
from model.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
fold = 0
epoch = 150
net_name = "c1"  # R50-ViT-B_16
train_dataset = "TN3K"  # TN3K DDTI
test_dataset = "TN3K"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    # Model settings
    parser.add_argument('-model_name', type=str, default=net_name)
    parser.add_argument('-epoch', type=str, default=epoch)
    parser.add_argument('-save_path', type=str, default=f"./weights/{net_name}/{train_dataset}/fold{fold}/model_{epoch}_best.pth")
    # Train settings
    parser.add_argument('-dataset', type=str, default=train_dataset)  # TN3K, DDTI
    parser.add_argument('-fold', type=int, default=fold)
    parser.add_argument('-input_size', type=int, default=256)
    return parser.parse_args()

def main(args):
    print(f"Model:{net_name}  fold:{fold}  epoch:{epoch}  dataset:{test_dataset}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if  'unet' == args.model_name:
        deploy_model = Unet(in_ch=3, out_ch=1)
    elif 'sgunet' == args.model_name:
        deploy_model = SGUNet()
    elif 'unext' == args.model_name:
        deploy_model = UNext(num_classes=1, img_size=256)
    elif 'cpfnet' == args.model_name:
        deploy_model = CPFNet()
    elif 'trfe' == args.model_name:
        deploy_model = TRFENet(3, 1)
    elif 'aaunet' == args.model_name:
        deploy_model = AAU_Net(in_channels=1, classes_num=2, channels_list=[32, 64, 128, 256, 512])
    elif 'ViT' in args.model_name:  # R50-ViT-B_16
        config_vit = CONFIGS_ViT_seg[args.model_name]
        config_vit.n_classes = 1
        config_vit.n_skip = 3  # 这里的n_skip含义不明,R50是3,别的是2?
        if args.model_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.input_size / 16), int(args.input_size / 16))
        deploy_model = ViT_seg(config_vit, img_size=args.input_size, num_classes=config_vit.n_classes)
    elif 'smaatunet' in args.model_name:
        deploy_model = SmaAt_UNet(3, 1)
    elif 'unext' == args.model_name:
        deploy_model = UNext(num_classes=1, img_size=256)
    elif 'unetpp' == args.model_name:
        deploy_model = UnetPlusPlus(in_ch=3, out_ch=1)
    elif 'attunet' == args.model_name:
        deploy_model = AttU_Net(in_channel=3, num_classes=1)
    elif 'deeplabv3plus' == args.model_name:
        deploy_model = deeplabv3plus_resnet101(num_classes=1, output_stride=16)
    elif 'dscaunet' == args.model_name:
        deploy_model = Model(3, 1)
    elif 'deeplabv3plus_8' == args.model_name:
        deploy_model = deeplabv3plus_resnet101(num_classes=1, output_stride=8)
    else:
        raise NotImplementedError
    deploy_model.load_state_dict(torch.load(args.save_path))
    deploy_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deploy_model.to(device)
    print("wnet have {}M paramerters in total".format(sum(x.numel() for x in deploy_model.parameters()) / 1e6))

    # 测试数据预处理
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    if args.dataset == 'DDTI':
        test_data = DDTI(mode='train', transform=composed_transforms_ts, return_size=False, fold=args.fold, test=True)
    elif args.dataset == 'TN3K':
        test_data = TN3K(mode='train', transform=composed_transforms_ts, return_size=False, fold=args.fold, test=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    # 存储特征图
    feature_maps = []

    # 注册钩子函数，用于存储特征图
    def hook_fn(module, input, output):
        feature_maps.append(output)

    layers_to_visualize = [getattr(deploy_model, 'conv1'), getattr(deploy_model, 'conv2'), getattr(deploy_model, 'conv3'), getattr(deploy_model, 'conv4'), getattr(deploy_model, 'conv5'), getattr(deploy_model, 'conv6'), getattr(deploy_model, 'conv7'), getattr(deploy_model, 'conv8'), getattr(deploy_model, 'conv9'), getattr(deploy_model, 'conv10')]  # 添加更多层

    # 注册钩子
    handles = []
    for layer in layers_to_visualize:
        handle = layer.register_forward_hook(hook_fn)
        handles.append(handle)

    def test_and_visualize_heatmap():
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                image = sample['image'].to(device)  # 获取输入图像
                label = sample['label'].to(device)  # 获取标签（如果需要）

                # 清空 feature_maps 以存储当前图像的特征图
                feature_maps.clear()

                # 将图像转换为 numpy 格式
                image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 转换为 HWC 格式
                image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))  # 归一化

                # 模型推理
                _ = deploy_model(image)  # 只需要调用模型，不需要保存输出

                # 创建一个大图，包含原始图像、标签和热力图

                # 计算行数和列数
                num_rows = 2
                num_cols = 6  # 确保列数不会超过6
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))  # 2行 + 特征图数量列

                # 显示原始图像
                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')  # 不显示坐标轴

                # 显示标签，使用 squeeze 去掉多余维度
                axes[1, 0].imshow(label.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)  # 处理多余的维度
                axes[1, 0].set_title("Label")
                axes[1, 0].axis('off')

                # 显示每个特征图
                for i, feature_map in enumerate(feature_maps):
                    feature_map = feature_map.squeeze(0).cpu().numpy()  # 获取特征图并移除批次维度
                    heatmap = np.mean(feature_map, axis=0)  # 使用通道平均值作为单通道热力图

                    # 归一化热力图
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) if np.max(
                        heatmap) > np.min(heatmap) else np.zeros_like(heatmap)

                    # 使用 jet 色彩映射
                    if i < 5:  # conv1到conv5是编码器层
                        axes[0, i + 1].imshow(heatmap, cmap='jet', interpolation='nearest')
                        axes[0, i + 1].set_title(f"Encoder Conv {i + 1}")
                        axes[0, i + 1].axis('off')  # 不显示坐标轴
                    else:  # conv6到conv10是解码器层
                        if i - 5 < num_cols - 1:  # 确保不会超出列数
                            axes[1, i - 5 + 1].imshow(heatmap, cmap='jet', interpolation='nearest')  # +1以便为标签留出空间
                            axes[1, i - 5 + 1].set_title(f"Decoder Conv {i + 1}")
                            axes[1, i - 5 + 1].axis('off')  # 不显示坐标轴

                # 删除多余的坐标轴
                for ax in axes.flat:
                    if not ax.has_data():  # 如果没有数据则删除
                        ax.remove()

                # 调整布局
                plt.tight_layout()
                plt.show()

                # 只生成第一个图像的热力图

                # break  # 如果想在处理多幅图像时只显示一幅图，请保留此行

    # 调用热力图可视化函数
    test_and_visualize_heatmap()

    # 清理钩子
    for handle in handles:
        handle.remove()
if __name__ == "__main__":
    args = get_arguments()
    main(args)
