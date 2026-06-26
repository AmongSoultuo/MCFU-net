import copy
import torch
from torch import nn
import torch.nn.functional as F


class hybrid_attention(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, h, w, deploy=False):
        super(hybrid_attention, self).__init__()
        self.channel_attention = channel_attention(in_channels=in_channels, out_channels=out_channel,
                                                   kernel_size=kernel_size, deploy=deploy)
        self.spatial_attention = spatial_attention(in_channels=out_channel, h=h, w=w)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class spatial_attention(nn.Module):
    def __init__(self, in_channels, h, w):
        super(spatial_attention, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_h = nn.Conv2d(in_channels=h, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(in_channels=w, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv_h(x.permute(0, 2, 1, 3))
        x_w = self.conv_w(x.permute(0, 3, 1, 2))
        x1 = x_h.permute(0, 2, 1, 3) * x_w.permute(0, 2, 3, 1)
        x2 = self.conv3x3(x)
        x11 = self.conv1x1(self.gap(x1)).reshape(b, -1, 1).permute(0, 2, 1)
        x12 = x2.reshape(b, c, -1)
        x21 = self.conv1x1(self.gap(x2)).reshape(b, -1, 1).permute(0, 2, 1)
        x22 = x1.reshape(b, c, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)
        return (x * weights.sigmoid()).reshape(b, c, h, w)


class channel_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, deploy=False):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 等效卷积核尺寸
        :param deploy: 是否设置为推理结构
        """
        super(channel_attention, self).__init__()
        self.rep_branch = RepParallel(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      deploy=deploy)
        self.conv_1x1_branch1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                        kernel_size=1,
                                                        bias=False)
                                              )
        self.conv_3x3_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                      bias=False))
        self.conv_1x1_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1,
                      bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x_rep = self.rep_branch(x)
        x_cat = torch.cat([self.conv_1x1_branch1(x_rep), self.conv_3x3_branch(x)], dim=1)

        return x_rep * self.conv_1x1_branch2(x_cat)


class fusion(nn.Module):
    def __init__(self, out_channels):
        """
        Ei与D(i-1)进行特征融合
        :param out_channels: 下采样的特征通道数
        """
        super(fusion, self).__init__()
        self.up_branch = nn.Sequential(nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels,
                                                 kernel_size=1, bias=False),
                                       nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                                          kernel_size=2, stride=2),
                                       nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                 kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(num_features=out_channels),
                                       nn.ReLU(inplace=True)
                                       )
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=1, bias=False),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(inplace=True)
                                     )
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(inplace=True)
                                     )

    def forward(self, x1, x2):
        x1 = self.up_branch(x1)
        x11 = x1 * self.branch1(x2) + x1
        return self.branch2(torch.cat([x11, x1], dim=1))


class RepParallel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, deploy=False):
        """
            创建一个结构重参数化block
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 最大卷积核尺寸
        :param deploy: 是否设置为推理结构
        """
        super().__init__()
        self.nonlinearity = nn.ReLU(inplace=True)
        self.deploy = deploy

        if kernel_size == 7:
            # (5, 3, 7)
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 1, 3]

        elif kernel_size == 5:
            # (3, 5)
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]

        if deploy:
            self.equivalent_kernel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=1, padding=kernel_size // 2, dilation=1, groups=1,
                                               bias=True)
        else:
            self.l_kernel = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size // 2, dilation=1, groups=1, bias=deploy)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('branch_k{}_{}'.format(k, r),
                                 conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1,
                                         padding=(r * (k - 1) + 1) // 2, dilation=r, groups=1,
                                         bias=False))

    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.equivalent_kernel(x))
        else:
            out = self.l_kernel(x)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('branch_k{}_{}'.format(k, r))
                out = out + conv(x)
            return self.nonlinearity(out)

    def _fuse_bn_tensor(self, branch: nn.Sequential):
        """
            将Conv与BN合并
        :param branch: Conv+BN
        :return: 等效的(w, b)
        """
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
            将模型转化为推理过程，创建等效核并删除原有训练结构
        """
        origin_k, origin_b = self._fuse_bn_tensor(self.l_kernel)
        self.equivalent_kernel = nn.Conv2d(in_channels=origin_k.size(0), out_channels=origin_k.size(0),
                                           kernel_size=origin_k.size(2), stride=1, padding=origin_k.size(2) // 2,
                                           dilation=1, groups=1, bias=True)
        for k, r in zip(self.kernel_sizes, self.dilates):
            branch = self.__getattr__('branch_k{}_{}'.format(k, r))
            branch_k, branch_b = self._fuse_bn_tensor(branch)
            origin_k = diliated_kerenl_into_large_kernel(large_kernel=origin_k, kernel=branch_k, dilate_rate=r)
            origin_b += branch_b
        self.equivalent_kernel.weight.data = origin_k
        self.equivalent_kernel.bias.data = origin_b
        self.__delattr__('l_kernel')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__('branch_k{}_{}'.format(k, r))


def conv_bn(in_channels, out_channels, kernel_size, stride, dilation, padding, groups=1, bias=False):
    """
        创建一个Conv+BN
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param dilation: 扩张率
    :param padding: 填充大小
    :param groups: 分组个数
    :param bias: 是否设置偏置项
    :return: 返回一个nn.Sequential(Conv+BN)
    """
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, dilation=dilation, bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn2(in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, groups=1, bias=False):
    """
        创建一个Conv+BN
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param dilation: 扩张率
    :param padding: 填充大小
    :param groups: 分组个数
    :param bias: 是否设置偏置项
    :return: 返回一个nn.Sequential(Conv+BN)
    """
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, dilation=dilation, bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('Relu', nn.ReLU(inplace=True))
    return result


def diliated_kerenl_into_large_kernel(kernel, dilate_rate, large_kernel):
    """
        将空洞卷积核合并到最大尺寸卷积核中
    :param kernel: 需合并的卷积核
    :param dilate_rate: 需合并的卷积核的扩张率
    :param large_kernel: 最大尺寸卷积核的权重
    :return: 合并后的卷积核权重
    """
    # 利用转置卷积将需填充的卷积核大小转到(k-1)*r+1
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        dilated = torch.cat(slices, dim=1)
    # 用0填充转到等效的卷积核大小
    rows_to_pad = large_kernel.size(2) // 2 - dilated.size(2) // 2
    equivalent_nondilated_kernel = F.pad(dilated, [rows_to_pad] * 4)

    return large_kernel + equivalent_nondilated_kernel


def repblock_convert(model, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
            module.deploy = True
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    x = torch.ones(1, 2, 10, 10)
    attention = hybrid_attention(in_channels=2, out_channel=5, kernel_size=5, deploy=False, h=10, w=10)
    out = attention(x)
    print(out.shape)
