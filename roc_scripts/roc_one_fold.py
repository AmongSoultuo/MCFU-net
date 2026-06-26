# 将单一模型的一个fold测得的preds值进行平均，并且保存至save_path，后续用于绘制roc均值曲线，默认取fold0

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def load_fold0(model_name, dataset_name, epoch):
    fold_path = f'./result/curve/{model_name}/{dataset_name}/fold0/{epoch}/'
    labels_path = os.path.join(fold_path, 'labels.csv')
    preds_path = os.path.join(fold_path, 'preds.csv')

    # 检查文件是否存在
    if not os.path.exists(labels_path) or not os.path.exists(preds_path):
        print("Labels or predictions CSV file not found in fold 0.")
        return None, None

    # 读取数据
    labels = np.loadtxt(labels_path, delimiter=',').astype(int)
    preds = np.loadtxt(preds_path, delimiter=',')

    return labels, preds


def plot_roc_curve(labels, preds):
    # 输出模型相关参数
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epoch: {epoch}")
    print(f"Fold: 0")

    # 计算 ROC 曲线的指标
    fpr, tpr, _ = roc_curve(labels, preds)

    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='r', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def save_results(labels, preds, model_name, dataset_name, epoch):
    save_path = f'./result/curve/{model_name}/{dataset_name}/{epoch}_avg/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savetxt(os.path.join(save_path, 'labels_avg.csv'), labels, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(save_path, 'preds_avg.csv'), preds, delimiter=",", fmt="%f")
    print(f"Labels and predictions saved to {save_path}")


# 使用示例
model_name = 'UNeXt'  # AttUnet DCSAU-Net MCFU-net(Ours) TransUnet Sgunet SmaAt-UNet U-net Unet++ ASPP-UNet UNeXt
dataset_name = 'TN3K'
epoch = '150'

labels, preds = load_fold0(model_name, dataset_name, epoch)

# 检查是否成功读取，绘制ROC曲线并且保存csv
if labels is not None and preds is not None:
    plot_roc_curve(labels, preds)
    save_results(labels, preds, model_name, dataset_name, epoch)
else:
    print("Failed to load fold 0 data.")