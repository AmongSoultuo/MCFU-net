# 基于roc_one_fold.py的结果，将单一模型的5个fold测得的preds值进行平均，并且保存至save_path，后续用于绘制roc均值曲线

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def average_labels_preds(model_name, dataset_name, epoch, folds=5):
    labels_all_folds = []
    preds_all_folds = []

    for i in range(folds):
        fold_path = f'./result/curve/{model_name}/{dataset_name}/fold{i}/{epoch}/'
        labels_path = os.path.join(fold_path, 'labels.csv')
        preds_path = os.path.join(fold_path, 'preds.csv')

        # 检查文件是否存在
        if not os.path.exists(labels_path) or not os.path.exists(preds_path):
            print(f"Labels or predictions CSV file not found in fold {i}.")
            return None, None

        # 读取数据并添加到列表中
        labels = np.loadtxt(labels_path, delimiter=',').astype(int)
        preds = np.loadtxt(preds_path, delimiter=',')

        labels_all_folds.append(labels)
        preds_all_folds.append(preds)

    # 将所有折叠的数组堆叠并取均值
    labels_mean = np.mean(np.stack(labels_all_folds), axis=0).astype(int)
    preds_mean = np.mean(np.stack(preds_all_folds), axis=0)

    return labels_mean, preds_mean


def plot_roc_curve(labels, preds):
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

def save_avg_results(labels, preds, model_name, dataset_name, epoch):
    save_path = f'./result/curve/{model_name}/{dataset_name}/{epoch}_avg/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savetxt(os.path.join(save_path, 'labels_avg.csv'), labels, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(save_path, 'preds_avg.csv'), preds, delimiter=",", fmt="%f")
    print(f"Averaged labels and predictions saved to {save_path}")


# 使用示例
model_name = 'dscaunet'  # R50-ViT-B_16 asppunet attunet dscaunet_new sgunet smaatunet unet unetpp MCFU-net(Ours)
dataset_name = 'TN3K'
epoch = '150'

labels_avg, preds_avg = average_labels_preds(model_name, dataset_name, epoch)

# # 检查是否成功读取并取均值，绘制ROC曲线并且保存均值csv
# if labels_avg is not None and preds_avg is not None:
#     plot_roc_curve(labels_avg, preds_avg)
#     save_avg_results(labels_avg, preds_avg, model_name, dataset_name, epoch)
# else:
#     print("Failed to calculate the average for labels or predictions.")

plot_roc_curve(labels_avg, preds_avg)
save_avg_results(labels_avg, preds_avg, model_name, dataset_name, epoch)