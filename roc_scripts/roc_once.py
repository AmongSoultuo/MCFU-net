# 显示单一模型的任一fold测得的roc曲线。对于绘制ROC无用，仅用于测试数据是否正常，足以支持roc曲线顺利生成

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_curves_from_csv(save_path):
    # 确保 CSV 文件存在
    labels_path = os.path.join(save_path, 'labels.csv')
    preds_path = os.path.join(save_path, 'preds.csv')
    if not os.path.exists(labels_path) or not os.path.exists(preds_path):
        print("Labels or predictions CSV file not found.")
        return

    # 读取数据
    labels = np.loadtxt(labels_path, delimiter=',').astype(int)
    preds = np.loadtxt(preds_path, delimiter=',')

    # 计算 ROC 和 PR 曲线的指标
    fpr, tpr, _ = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)

    # 绘制 PR 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='r', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

# 使用示例
model_name = 'unet'  # 替换为实际模型名称
dataset_name = 'DDTI'  # 替换为实际数据集名称
epoch = '100'  # 替换为实际 epoch 数
fold = '0'  # 替换为实际 fold 数

save_path = f'./result/curve/{model_name}/{dataset_name}/fold{fold}/{epoch}/'
plot_curves_from_csv(save_path)
