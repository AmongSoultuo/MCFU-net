import argparse
import copy
import math
import os
import time
import random
import torch.optim.lr_scheduler as lr_scheduler
# PyTorch includes
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
# Dataloaders includes
from dataloaders import custom_transforms as trforms
from dataloaders.tn3k import TN3K
from dataloaders.ddti import DDTI
from torchvision import transforms
# Model includes
from model.Unet import Unet
from model.Unetpp import UnetPlusPlus
from model.AAU_Net import AAU_Net
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
from model.ASPPUnet import ASPPUnet
from metrics import evaluate, plot_curves, plot_loss

from MCFU_net import MCFU_net

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='1')
    # Model settings
    parser.add_argument('-model_name', type=str, default='MCFU-net')
    parser.add_argument('-fold', type=int, default=0)
    parser.add_argument("--rep", default=False, type=bool)
    # Train settings
    parser.add_argument('-dataset', type=str, default='DDTI')  # TN3K, DDTI
    parser.add_argument('-input_size', type=int, default=256)
    parser.add_argument('-batch_size', type=int, default=6)
    parser.add_argument('-num_epoch', type=int, default=150)
    parser.add_argument("--warm-up-epochs", default=5, type=int)
    # Optimizer settings
    parser.add_argument('-lr', type=float, default=1e-4)
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

def main(args):
    print(f"Model:{args.model_name}  fold:{args.fold}  epoch:{args.num_epoch}  dataset:{args.dataset}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if 'MCFU-net' == args.model_name:
        net = MCFU_net(in_ch=3, out_ch=1)
    # others
    elif 'unet' == args.model_name:
        net = Unet(in_ch=3, out_ch=1)
    elif 'aaunet' == args.model_name:
        net = AAU_Net(in_channels=1, classes_num=2, channels_list=[32, 64, 128, 256, 512])
    elif 'sgunet' == args.model_name:
        net = SGUNet()
    elif 'unext' == args.model_name:
        net = UNext(num_classes=1, img_size=256)
    elif 'cpfnet' == args.model_name:
        net = CPFNet()
    elif 'trfe' == args.model_name:
        net = TRFENet(3, 1)
    elif 'ViT' in args.model_name:  # R50-ViT-B_16
        config_vit = CONFIGS_ViT_seg[args.model_name]
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        if args.model_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.input_size / 16), int(args.input_size / 16))
        net = ViT_seg(config_vit, img_size=args.input_size, num_classes=config_vit.n_classes)
    elif 'smaatunet' in args.model_name:
        net = SmaAt_UNet(3, 1)
    elif 'unetpp' == args.model_name:
        net = UnetPlusPlus(in_ch=3, out_ch=1)
    elif 'deeplabv3plus_50' == args.model_name:
        net = deeplabv3plus_resnet50(num_classes=1, output_stride=16)
    elif 'dscaunet' == args.model_name:
        net = Model(3, 1)
    elif 'dscaunet_new' == args.model_name:
        net = Model(3, 1)
    elif 'attunet' == args.model_name:
        net = AttU_Net(in_channel=3, num_classes=1)
    elif 'deeplabv3plus_new' == args.model_name:
        net = deeplabv3plus_resnet50(num_classes=1, output_stride=16)
    elif 'asppunet' == args.model_name:
        net =ASPPUnet(3, 1)

    else:
        raise NotImplementedError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)
    # warm_up_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
    #         math.cos((epoch - args.warm_up_epochs) / (args.num_epoch - args.warm_up_epochs) * math.pi) + 1)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_cosine_lr)

    # 定义图像转换
    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(int(args.input_size), int(args.input_size))),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    if args.dataset == 'DDTI':
        train_data = DDTI(mode='train', transform=composed_transforms_tr, return_size=False, fold=args.fold)
        val_data = DDTI(mode='val', transform=composed_transforms_ts, return_size=False, fold=args.fold)
    elif args.dataset == 'TN3K':
        train_data = TN3K(mode='train', transform=composed_transforms_tr, return_size=False, fold=args.fold)
        val_data = TN3K(mode='val', transform=composed_transforms_ts, return_size=False, fold=args.fold)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    train_losses = []
    val_losses = []
    best_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'specificity': 0.0,
        'accuracy': 0.0,
        'iou': 0.0,
        'dice': 0.0,
        'mae': 0.0
    }

    # 打印表头
    print(
        '{:<7}\t{:<10}\t{:<10}\t{:<8}\t{:<10}\t{:<10}\t{:<12}\t{:<10}\t{:<8}\t{:<8}\t{:<8}'.format(
            'Epoch', 'Train_Loss', 'Val_Loss', 'Time', 'Precision', 'Recall', 'Specificity',
            'Accuracy', 'IoU', 'DICE', 'MAE'))
    path = f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}'
    if not os.path.exists(path):
        os.makedirs(path)
    for epoch in range(args.num_epoch):
        best_iou = best_metrics['iou']
        start_time = time.time()
        running_loss_train = 0.0
        running_loss_val = 0.0
        net.train()
        for i, sample_batched in enumerate(train_loader):
            images, labels = sample_batched['image'], sample_batched['label']
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.float()

            # labels = labels.squeeze(1).long()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        train_loss = running_loss_train / len(train_loader.dataset)  # 计算平均训练损失
        train_losses.append(train_loss)  # 记录所有训练损失

        # current_lr = optimizer.param_groups[0]['lr']

        # scheduler.step()
        net.eval()
        deploy_model = copy.deepcopy(net)
        if args.rep:
            deploy_model = repblock_convert(deploy_model, do_copy=False)
        preds = []
        labels_all = []
        for sample_batched in test_loader:
            images, labels = sample_batched['image'], sample_batched['label']
            images, labels = images.to(device), labels.to(device)
            outputs = deploy_model(images)
            labels = labels.float()
            pred = (outputs > 0.5).float()
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()
            preds.append(pred)
            labels_all.append(labels)
        preds_all = torch.cat(preds, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        val_loss = running_loss_val / len(test_loader.dataset)
        val_losses.append(val_loss)
        precision, recall, specificity, accuracy, iou, dice, mae = evaluate(epoch, preds_all, labels_all,
                                                                            f'./result/metrics/{args.model_name}_fold{args.fold}_7_{args.dataset}.csv')

        # 打印每轮结果
        print('{:<7}\t{:<10.4f}\t{:<10.4f}\t{:<8.2f}\t{:<10.4f}\t{:<10.4f}\t{:<12.4f}\t'
              '{:<10.4f}\t{:<8.4f}\t{:<8.4f}\t{:<8.4f}'
              .format(epoch + 1, train_loss, val_loss, epoch_time, precision.item(), recall.item(), specificity.item(),
                      accuracy.item(), iou.item(), dice.item(), mae.item()), end="")
        # plot_curves(labels_all, preds_all, epoch, f'./result/curve/{args.model_name}/{args.dataset}')  # 保存ROC曲线和PR曲线

        # 保存loss曲线
        # torch.save(net.state_dict(), f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_{epoch+1}.pth')  # 保存模型

        # 比较评价指标并保存最好的模型
        if iou.item() > best_iou:
            best_metrics = {
                'precision': precision.item(),
                'recall': recall.item(),
                'specificity': specificity.item(),
                'accuracy': accuracy.item(),
                'iou': iou.item(),
                'dice': dice.item(),
                'mae': mae.item()
            }
            print(f"                    Now best model：{epoch + 1}")

            # torch.save(deploy_model.state_dict(),
            #            f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_150_best.pth')
            if epoch + 1 > 150:
                torch.save(deploy_model.state_dict(),
                           f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_200_best.pth')  # 保存表现最好的模型
            elif epoch + 1 <= 150 and epoch + 1 > 100:
                torch.save(deploy_model.state_dict(),
                           f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_150_best.pth')
            elif epoch + 1 <= 100 and epoch + 1 > 50:
                torch.save(deploy_model.state_dict(),
                           f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_100_best.pth')
            else:
                torch.save(deploy_model.state_dict(),
                           f'./weights/{args.model_name}/{args.dataset}/fold{args.fold}/model_50_best.pth')
        else:
            print("                    This round's performance is not as good as the previous one.")
    plot_loss(train_losses, val_losses, 150, f'./result/loss/{args.model_name}/{args.dataset}/{args.fold}')
if __name__ == "__main__":
    args = get_arguments()
    main(args)
