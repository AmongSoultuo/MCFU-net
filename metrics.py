import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.ndimage import distance_transform_edt as edt


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
                pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        # print(right_hd, ' ', left_hd)

        return torch.max(right_hd, left_hd)


hd_metric = HausdorffDistance()


def evaluate(epoch, pred, gt, file_path=None):
    """
    进行评价指标的运算，并且将结果保存到csv文件中和返回
    :param epoch: 当前轮数
    :param pred: 预测图
    :param gt: mask
    :return: 返回评价指标的运算结果
    """
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = pred.float()
    gt_binary = (gt >= 0.5).float()

    TP = (pred_binary * gt_binary).sum()
    FP = ((pred_binary == 1) & (gt_binary == 0)).sum()
    TN = ((pred_binary == 0) & (gt_binary == 0)).sum()
    FN = ((pred_binary == 0) & (gt_binary == 1)).sum()

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU = TP / (TP + FP + FN)
    MAE = torch.abs(pred - gt).mean()
    DICE = 2 * IoU / (IoU + 1)

    if file_path != None:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(
                    ['Epoch', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'IoU', 'DICE',
                     'MAE'])
            writer.writerow([epoch+1, f'{Precision.item():.4f}', f'{Recall.item():.4f}', f'{Specificity.item():.4f}',
                             f'{accuracy.item():.4f}', f'{IoU.item():.4f}',
                             f'{DICE.item():.4f}', f'{MAE.item():.4f}'])

    return Precision, Recall, Specificity, accuracy, IoU, DICE, MAE


def calculate_hd95_for_dataset(pred_list, gt_list, device='cuda'):
    """
    计算整个验证集的 95% Hausdorff 距离。

    参数:
    - pred_list: 预测的二值分割图像列表 (list of PyTorch tensors)。
    - gt_list: 实际的二值分割标签列表 (list of PyTorch tensors)。
    - device: 计算设备，'cuda' 或 'cpu'。

    返回:
    - average_hd95: 验证集的平均 95% Hausdorff 距离。
    """

    hd95_values = []

    for i in range(pred_list.shape[0]):
        hd95_value = hd_metric.compute(pred_list[i].unsqueeze(0), gt_list[i].unsqueeze(0))
        hd95_values.append(hd95_value)

    average_hd95 = np.mean(hd95_values)

    return average_hd95


def evaluate_hd95(pred, gt):
    """
    进行评价指标的运算（返回HD95版本，不含CSV保存）
    用于 test.py / test_all_fold.py 等测试脚本
    :param pred: 预测图列表
    :param gt: mask 列表
    :return: Precision, Recall, Specificity, accuracy, IoU, DICE, HD95
    """
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = [p.float() for p in pred]
    gt_binary = [(g >= 0.5).float() for g in gt]

    TP = sum((p * g).sum() for p, g in zip(pred_binary, gt_binary))
    FP = sum(((p == 1) & (g == 0)).sum() for p, g in zip(pred_binary, gt_binary))
    TN = sum(((p == 0) & (g == 0)).sum() for p, g in zip(pred_binary, gt_binary))
    FN = sum(((p == 0) & (g == 1)).sum() for p, g in zip(pred_binary, gt_binary))

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU = TP / (TP + FP + FN)
    DICE = 2 * IoU / (IoU + 1)

    HD95 = calculate_hd95_for_dataset(pred, gt)

    return Precision, Recall, Specificity, accuracy, IoU, DICE, HD95


# def plot_curves(labels_all, preds_all, epoch, save_path):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     labels_fl = labels_all.detach().cpu().numpy().flatten().astype(int)  # 转换为整数类型
#     preds_fl = preds_all.detach().cpu().numpy().flatten().astype(int)
#
#     fpr, tpr, _ = roc_curve(labels_fl, preds_fl)
#     precision, recall, _ = precision_recall_curve(labels_fl, preds_fl)
#
#     # 绘制 PR 曲线
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(recall, precision, color='b', label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc='best')
#
#     # 绘制 ROC 曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(fpr, tpr, color='r', label='ROC curve')
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc='best')
#
#     plt.tight_layout()
#
#     # 保存曲线图
#     plt.savefig(os.path.join(save_path + f'/epoch_{epoch}_curves.png'))
#     plt.close()

def plot_curves(labels_all, preds_all, epoch, save_path):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 转换数据为整数类型并保存
    labels_fl = labels_all.detach().cpu().numpy().flatten().astype(int)
    preds_fl = preds_all.detach().cpu().numpy().flatten()

    # 保存原始数据到CSV文件
    np.savetxt(os.path.join(save_path, f'labels.csv'), labels_fl, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(save_path, f'preds.csv'), preds_fl, delimiter=",", fmt="%f")

    # 计算ROC和PR曲线的指标
    fpr, tpr, _ = roc_curve(labels_fl, preds_fl)
    precision, recall, _ = precision_recall_curve(labels_fl, preds_fl)

    # 绘制 PR 曲线并单独保存
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_path, 'pr_curve.png'))
    plt.close()

    # 绘制 ROC 曲线并单独保存
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='r', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()


def plot_loss(train_losses, val_losses, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 将 train_loss 和 val_loss 数据保存到文本文件
    with open(os.path.join(save_path, f'{epoch}.txt'), 'w') as file:
        file.write("Epoch\tTrain Loss\tVal Loss\n")
        for i in range(epoch):
            file.write(f"{i + 1}\t{train_losses[i]:.6f}\t{val_losses[i]:.6f}\n")

    if epoch % 200 == 0:
        epochs = range(1, epoch + 1)

        # 保存训练损失曲线图像
        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'train_{epoch}.png'))

        # 保存验证损失曲线图像
        plt.figure()
        plt.plot(epochs, val_losses, label='Val Loss', color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'val_{epoch}.png'))


if __name__ == "__main__":
    epoch = 9

    # 随机生成两个形状为（1，1，224，224）的output和label
    output = torch.rand(1, 1, 224, 224)
    label = torch.randint(0, 2, (1, 1, 224, 224)).float()

    # 调用评价指标函数进行验证
    Precision, Recall, Specificity, accuracy, IoU, DICE, MAE = evaluate(epoch, output, label)

    # 绘制曲线并保存
    plot_curves(epoch, label, output)

    root_path = './result/metrics'


