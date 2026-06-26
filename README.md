# MCFU-net 甲状腺超声图像分割项目

[![Paper](https://img.shields.io/badge/Paper-Information_2025-2ea44f?logo=mdpi)](./MCFU-net_paper.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.3390/info16111013-blue)](https://doi.org/10.3390/info16111013)

## 项目简介

本项目是论文 *Prediction Multiscale Cross-Level Fusion U-Net with Combined Wavelet Convolutions for Thyroid Nodule Segmentation*（Information 2025）的官方实现。

MCFU-net 提出了一种融合多分支小波卷积（MBWC）、尺度选择空洞金字塔（SSAP）与跨级特征融合（CLFM）的 U-Net 架构，在 **TN3K**（Dice 85.22%）和 **DDTI**（Dice 78.21%）两个公开数据集上均取得了较好的性能。本仓库提供完整的训练/测试代码、对比模型实现及评估工具。

---

## 项目结构

```
mcfunet/
├── data/                     # 数据集
├── dataloaders/              # 数据加载器（DDTI / TN3K）
├── model/                    # 对比模型
├── result/                   # 测试结果（运行 train.py / test.py 自动生成子目录）
├── roc_scripts/              # ROC曲线绘制脚本
│   ├── roc.py                # 多模型对比图（最终图）
│   ├── roc_all_fold.py       # 5-fold均值计算
│   ├── roc_once.py           # 单fold数据验证
│   └── roc_one_fold.py       # fold0均值计算
├── weights/                  # 训练权重
├── WTCONV/                   # 小波卷积模块
├── experiments.py            # 环境版本信息打印
├── feature_view.py           # 独立版特征图可视化
├── flop_param.py             # FLOPs 与参数量计算
├── fps.py                    # GPU 推理速度 FPS 测试
├── MCFU_net.py               # MCFU-net
├── MCFU-net_paper.pdf        # 论文全文
├── metrics.py                # 评估指标与可视化
├── README.md
├── requirements.txt          # Python 依赖
├── test.py                   # 测试脚本
├── test_all_fold.py          # 多fold指标统计（均值 ± 标准差）
├── test_demo.py              # 单图快速测试
├── train.py                  # 训练脚本（主）
├── train_parallel.py         # 训练脚本（GPU 低利用率时并行训练第二个模型）
```

---
## 快速开始

### 环境安装

```bash
# 1. 创建虚拟环境
conda create -n mcfunet python=3.9 -y && conda activate mcfunet

# 2. 安装 PyTorch（CUDA 11.8）
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其余依赖
pip install -r requirements.txt
```

### 数据准备

本项目不包含数据集和部分预训练模型，请从如下链接下载后放置到对应目录：

| 下载内容 | 下载地址 | 放置位置 | 说明 |
|----------|-------------|----------|------|
| 数据集 | **[百度网盘链接A](https://pan.baidu.com/s/114fEE2OEKncTosW7ibbjsw?pwd=8888)** | `data/` | TN3K 为已预处理数据；DDTI 含原始数据与预处理后数据，共约 312 MB |
| TransUNet 预训练模型 | **[百度网盘链接B](https://pan.baidu.com/s/1JRTw9q84Yk3ShxEG8Kf-Yw?pwd=6666)** | `model/transunet/` | `imagenet21k_R50+ViT-B_16.npz`，约 440 MB |

### 模型测试

> ⚠️ **注意**：`result/` 下的四个子目录（`curve/`、`loss/`、`metrics/`、`pre_mask/`）无需手动创建，运行 `train.py` 和 `test.py` 时会自动生成。

仓库已包含 MCFU-net 在 TN3K 数据集 fold0 的最佳权重，下载数据集后可直接测试：

```bash
# 建议单张图片测试，快速验证效果
python test_demo.py --model_name MCFU-net --dataset TN3K --fold 0 --epoch 150
```
预测结果将保存在 `test_one_mask/` 目录下。

```bash
# 完整测试集评估
python test.py -model_name MCFU-net -dataset TN3K -fold 0 -epoch 150
```

### 模型训练

```bash
# 主训练脚本
python train.py -gpu 0 -model_name MCFU-net -dataset TN3K -fold 0 -num_epoch 150

# 并行训练（与主训练同时跑另一个模型）
python train_parallel.py -gpu 1 -model_name unet -dataset DDTI -fold 0 -num_epoch 150
```

### ROC 曲线

```bash
# 1. 每个fold单独跑，生成roc绘图相关数据,放在result/curve中
python roc_scripts/roc_one_fold.py

# 2. 基于5个fold，生成fold均值数据
python roc_scripts/roc_all_fold.py

# 3. 读取所有模型均值数据，画最终多模型对比图
python roc_scripts/roc.py
```
---

## 核心文件详解

### MCFU_net.py

MCFU-net 核心模型，采用编码器-解码器结构：

| 模块 | 说明 |
|------|------|
| **DoubleConv1** | MBWC × 2）|
| **SSAP** | 尺度选择空洞金字塔模块 |
| **MBWC** | 多分支小波卷积模块 |
| **CLFM1/2/3** | 跨级融合模块 |

```python
from MCFU_net import MCFU_net
model = MCFU_net(in_ch=3, out_ch=1)
```

### metrics.py

| 函数 | 签名 | 返回 | 调用者 |
|------|------|------|--------|
| `evaluate(epoch, pred, gt, file_path)` | 含 CSV 保存 | Precision / Recall / Specificity / Accuracy / IoU / DICE / **MAE** | train.py, train_parallel.py |
| `evaluate_hd95(pred, gt)` | 不含 CSV | 同上但返回 **HD95** 替代 MAE | test.py, test_all_fold.py |
| `plot_curves(labels, preds, epoch, save_path)` | 保存 labels.csv + preds.csv + PR/ROC 图 | — | test.py, test_all_fold.py |
| `plot_loss(train_losses, val_losses, epoch, save_path)` | 保存 loss txt + 曲线图 | — | train.py, train_parallel.py |

### train.py / train_parallel.py

共用参数一览：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-gpu` | GPU 设备号 | `0` / `1` |
| `-model_name` | 模型名称 | `MCFU-net` |
| `-dataset` | 数据集（DDTI / TN3K） | `DDTI` |
| `-fold` | 交叉验证折数（0-4） | `0` |
| `-batch_size` | 批次大小 | `6` |
| `-num_epoch` | 训练轮数 | `150` |
| `-lr` | 学习率 | `1e-4` |

支持的模型名称：`MCFU-net` / `unet` / `unetpp` / `attunet` / `sgunet` / `unext` / `cpfnet` / `trfe` / `smaatunet` / `dscaunet` / `deeplabv3plus_50` / `asppunet` / `R50-ViT-B_16` 等。

`train_parallel.py` 参数与功能同 `train.py`，用于在 GPU 利用率低时同时训练第二个模型以缩短总训练时间。

### test.py

加载训练好的权重，在测试集上批量推理，输出：

- 预测 mask 图 → `result/pre_mask/{model}/{dataset}/fold{fold}/{epoch}/`
- ROC / PR 曲线 → `result/curve/{model}/{dataset}/fold{fold}/{epoch}/`
- 终端打印评估指标（Precision / Recall / Specificity / Accuracy / IoU / DICE / HD95）

### test_all_fold.py

遍历模型指定 epoch 下全部 5 个 fold 的权重，分别测试并计算指标，最终输出各指标的均值 ± 标准差。与 `test.py` 参数一致，区别如下：

| | test.py | test_all_fold.py |
|------|---------|------------------|
| 测试范围 | 单个 fold | 全部 5 个 fold |
| 输出 | 单 fold 指标 | 5-fold 均值 ± 标准差 |
| 额外依赖 | `plot_curves` | `plot_curves` + `evaluate_hd95` |

### test_demo.py

单张图片快速验证，输出原图 + GT + 预测的三列对比图并保存至 `test_one_mask/`。

---

## 引用 Citation

如果您觉得这项工作对您有帮助，请引用我们的论文。论文全文可在本仓库查看：[MCFU-net_paper.pdf](./MCFU-net_paper.pdf)

**论文主页**：[https://www.mdpi.com/2078-2489/16/11/1013](https://www.mdpi.com/2078-2489/16/11/1013)

### BibTeX

```bibtex
@Article{info16111013,
  author   = {Liu, Shengzhi and Tang, Haotian and Zhao, Junhao and Liu, Rundong and
              Zheng, Sirui and Hou, Kaiyao and Zhang, Xiyu and Liu, Fuyong and Ding, Chen},
  title    = {Prediction Multiscale Cross-Level Fusion U-Net with Combined Wavelet
              Convolutions for Thyroid Nodule Segmentation},
  journal  = {Information},
  volume   = {16},
  year     = {2025},
  number   = {11},
  article-number = {1013},
  issn     = {2078-2489},
  doi      = {10.3390/info16111013},
}
```

### Plain Text

```text
Liu, S.; Tang, H.; Zhao, J.; Liu, R.; Zheng, S.; Hou, K.; Zhang, X.; Liu, F.; Ding, C. Prediction Multiscale Cross-Level Fusion U-Net with Combined Wavelet Convolutions for Thyroid Nodule Segmentation. Information 2025, 16, 1013. https://doi.org/10.3390/info16111013
```
