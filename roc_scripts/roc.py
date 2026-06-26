# 用于最终ROC曲线的绘制

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_multiple_roc_curves_with_auc(model_list, dataset_name, epoch):
    auc_dict = {}  # 存储模型名称和 AUC 值

    # 计算每个模型的 ROC 曲线和 AUC 值，并存入字典
    for model_name in model_list:
        save_path = f'./result/curve/{model_name}/{dataset_name}/{epoch}_avg/'
        labels_path = os.path.join(save_path, 'labels_avg.csv')
        preds_path = os.path.join(save_path, 'preds_avg.csv')

        if not os.path.exists(labels_path) or not os.path.exists(preds_path):
            print(f"Labels or predictions CSV file not found for model {model_name}.")
            continue

        # 读取数据
        labels = np.loadtxt(labels_path, delimiter=',').astype(int)
        preds = np.loadtxt(preds_path, delimiter=',')

        # 计算 ROC 曲线和 AUC
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_score = roc_auc_score(labels, preds)
        auc_dict[model_name] = (fpr, tpr, auc_score)

    # 将模型按 AUC 从大到小排序
    sorted_auc = sorted(auc_dict.items(), key=lambda x: x[1][2], reverse=True)

    # 绘制曲线，按 AUC 从大到小的顺序
    for model_name, (fpr, tpr, auc_score) in sorted_auc:
        # 如果是 'MCFU-net(Ours)' 使用红色线条
        if model_name == 'MCFU-net(Ours)':
            plt.plot(fpr, tpr, linestyle='-', color='red', label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=1.5)
        else:
            plt.plot(fpr, tpr, linestyle='--', label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=1.5)

    # 自定义缩放比例，调整纵轴范围和刻度
    plt.xlim([0.00, 1.00])  # 可调节范围
    plt.ylim([0.75, 1.00])  # 设置纵轴起始坐标
    plt.xticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])  # 自定义横轴刻度
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])  # 自定义纵轴刻度

    # 图像标签
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves of {dataset_name}')
    plt.legend(loc='best')
    plt.savefig(f"{dataset_name}", dpi=1200)
    plt.show()

model_list = ['AttUnet', 'DCSAU-Net', 'MCFU-net(Ours)', 'TransUnet', 'Sgunet', 'SmaAt-UNet', 'U-net', 'Unet++', 'ASPP-UNet']
dataset_name = 'TN3K'  # DDTI TN3K
epoch = '150'

plot_multiple_roc_curves_with_auc(model_list, dataset_name, epoch)
