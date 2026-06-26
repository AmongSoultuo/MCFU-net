# 对同一模型的5折fold训练出来的5个权重进行测试
import argparse
import copy
import os
import time
from torchstat import stat
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
from model.utils import *
from model.sgunet import SGUNet
from model.cpfnet import CPFNet
from model.trfe import TRFENet
from model.SmaAtunet.SmaAt_UNet import SmaAt_UNet
from model.attunet import AttU_Net
from model.unext import UNext
from model.deeplab.modeling import deeplabv3plus_resnet101
from model.deeplab_50.modeling import deeplabv3plus_resnet50
from model.dscaunet.DCSAU_Net import Model
from model.unext import UNext
from model.ASPPUnet import ASPPUnet
from MCFU_net import MCFU_net
from metrics import evaluate_hd95
import numpy as np
from model.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

fold = 1
epoch = 150
epoch2 = [150, 150, 150, 150, 150]
net_name = "unetpp"

# R50-ViT-B_16 MCFU-net deeplabv3plus dscaunet smaatunet sgunet unet unetpp attunet unext asppunet

# train_dataset = "TN3K"
# test_dataset = "TN3K"
train_dataset = "DDTI"
test_dataset = "DDTI"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    # Model settings
    parser.add_argument('-model_name', type=str, default=net_name)
    parser.add_argument('-epoch', type=str, default=epoch)
    parser.add_argument('-save_path', type=str, default=f"./weights/{net_name}/{train_dataset}/fold{fold}/model_{epoch}_best.pth")
    # Train settings
    parser.add_argument('-dataset', type=str, default=test_dataset)  # TN3K, DDTI
    parser.add_argument('-fold', type=int, default=fold)
    parser.add_argument('-input_size', type=int, default=256)
    return parser.parse_args()
import numpy as np

def calculate_mean_std(metrics_list):
    """
    计算5折fold的指标的平均值和标准差
    """
    metrics_list_cpu = [
        [metric.cpu().numpy() if isinstance(metric, torch.Tensor) else metric for metric in metrics]
        for metrics in metrics_list
    ]
    metrics_array = np.array(metrics_list_cpu)
    mean_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)
    return mean_metrics, std_metrics

def print_metrics(metrics, fold_num=None):
    if fold_num is not None:
        print(f"Fold {fold_num + 1}:")
    else:
        print("Average Metrics:")
    metric_names = ["Precision", "Recall", "Specificity", "Accuracy", "IoU", "DICE", "HD95"]
    for i, name in enumerate(metric_names):
        if fold_num is not None:
            if name == "HD95":
                print(f"{name}: {metrics[i]:.2f}")
            else:
                print(f"{name}: {metrics[i] * 100:.2f}")
        else:
            if name == "HD95":
                print(f"{name}: {metrics[i][0]:.2f} {metrics[i][1]:.2f}")
            else:
                print(f"{name}: {metrics[i][0] * 100:.2f} {metrics[i][1] * 100:.2f}")
    print("\n")


def main(args):
    if 'unet' == args.model_name:
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
        config_vit.n_skip = 3
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
    elif 'asppunet' == args.model_name:
        deploy_model = ASPPUnet(3, 1)
    elif 'deeplabv3plus_50' == args.model_name:
        deploy_model = deeplabv3plus_resnet50(num_classes=1, output_stride=16)
    elif 'deeplabv3plus' == args.model_name:
        deploy_model = deeplabv3plus_resnet101(num_classes=1, output_stride=16)
    elif 'dscaunet' == args.model_name:
        deploy_model = Model(3, 1)
    elif 'MCFU-net' == args.model_name:
        deploy_model = MCFU_net(in_ch=3, out_ch=1)
    else:
        raise NotImplementedError
    print(f"Model:{net_name}  fold:{fold}  epoch:{epoch}  dataset:{test_dataset}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 测试数据预处理
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    if args.dataset == 'DDTI':
        test_data = DDTI(mode='test', transform=composed_transforms_ts, return_size=False, fold=args.fold, test=True)
    elif args.dataset == 'TN3K':
        test_data = TN3K(mode='test', transform=composed_transforms_ts, return_size=False, fold=args.fold, test=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    # print('{:<7}\t{:<10}\t{:<10}\t{:<12}\t{:<10}\t{:<8}\t{:<8}\t{:<8}'.format(
    #         'Epoch', 'Precision', 'Recall', 'Specificity',

    #         'Accuracy', 'IoU', 'DICE', 'MAE'))
    save_dir = f'./result/pre_mask/{args.model_name}/{args.dataset}/fold{args.fold}/{args.epoch}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        metrics_folds = []
        for i in range(5):
            save_path = f"./weights/{net_name}/{train_dataset}/fold{i}/model_{epoch2[i]}_best.pth"
            deploy_model.load_state_dict(torch.load(save_path))
            deploy_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            deploy_model.to(device)
            preds = []
            preds_curve = []
            labels_all = []
            for sample_batched in test_loader:
                images, labels, names, sizes = sample_batched['image'], sample_batched['label'], sample_batched.get(
                    'label_name'), sample_batched['size']
                images, labels = images.to(device), labels.to(device)
                outputs = deploy_model(images)
                labels = labels.float()
                pred = (outputs > 0.5).float()
                preds.append(pred)
                preds_curve.append(outputs)
                labels_all.append(labels)

                # # pre_mask
                # shape = (sizes[0, 0], sizes[0, 1])
                # prob_pred = F.interpolate(outputs, size=shape, mode='bilinear', align_corners=True).cpu().data
                # save_data = prob_pred[0]
                # save_png = save_data[0].numpy()
                # save_saliency = save_png * 255
                # save_saliency = save_saliency.astype(np.uint8)
                #
                # save_png = np.round(save_png)
                # # print(save_png.shape)
                #
                # save_png = save_png * 255
                # save_png = save_png.astype(np.uint8)
                # save_path = save_dir + names[0]
                # if not os.path.exists(save_path[:save_path.rfind('/')]):
                #     os.makedirs(save_path[:save_path.rfind('/')])
                # save_path_s = save_dir + 's' + names[0]
                # cv2.imwrite(save_path_s, save_saliency)
                # cv2.imwrite(save_dir + names[0], save_png)
            preds_all = torch.cat(preds, dim=0)
            preds_curve_all = torch.cat(preds_curve, dim=0)
            labels_all = torch.cat(labels_all, dim=0)
            plot_curves(labels_all, preds_curve_all, epoch,f'./result/curve/{args.model_name}/{args.dataset}/fold{i}/{args.epoch}/')
            metrics = evaluate_hd95(preds_all, labels_all)

            # 存储当前折fold的指标
            metrics_folds.append(metrics)

            # 打印当前折fold的指标
            print_metrics(metrics, fold_num=i)
        
        # 计算5折fold的指标的平均值和标准差
        mean_metrics, std_metrics = calculate_mean_std(metrics_folds)

        # 计算5折fold的指标的平均值和标准差的组合
        combined_metrics = list(zip(mean_metrics, std_metrics))

        # 打印5折fold的指标的平均值和标准差的组合
        print_metrics(combined_metrics)
if __name__ == "__main__":
    args = get_arguments()
    main(args)
