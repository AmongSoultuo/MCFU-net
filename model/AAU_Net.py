import torch
import torch.nn as nn


class HAAM(nn.Module):
    def __init__(self, channel_nums, out_channel):
        super(HAAM, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel_nums, 2 * channel_nums, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(2 * channel_nums),
                                    nn.ReLU(),
                                    nn.Conv2d(2 * channel_nums, 2 * channel_nums, kernel_size=1),
                                    nn.BatchNorm2d(2 * channel_nums),
                                    nn.ReLU())

        # Channel attention
        self.conv_3x3 = nn.Sequential(nn.Conv2d(channel_nums, 2 * channel_nums, kernel_size=3, padding=3, dilation=3),
                                      nn.BatchNorm2d(2 * channel_nums),
                                      nn.ReLU())

        self.conv_5x5 = nn.Sequential(nn.Conv2d(channel_nums, 2 * channel_nums, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(2 * channel_nums),
                                      nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcn = nn.Sequential(nn.Linear(2 * channel_nums, channel_nums),
                                 nn.BatchNorm1d(channel_nums),
                                 nn.ReLU(),
                                 nn.Linear(channel_nums, 2 * channel_nums),
                                 nn.Sigmoid())

        self.conv_1x1 = nn.Sequential(nn.Conv2d(2 * channel_nums, 2 * channel_nums, kernel_size=1),
                                      nn.BatchNorm2d(2 * channel_nums),
                                      nn.ReLU())

        # Spatial attention
        self.relu = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(2 * channel_nums, 1, kernel_size=1),
                                  nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(2 * channel_nums, out_channel, kernel_size=1),
                                  nn.BatchNorm2d(out_channel))

    def forward(self, x):
        b, c, _, _ = x.size()
        spatil_data = self.conv_1(x)

        # Channel attention
        conv_3x3 = self.conv_3x3(x)
        conv_5x5 = self.conv_5x5(x)

        data1 = torch.add(conv_3x3, conv_5x5)
        data1 = self.avg_pool(data1).view(b, 2 * c)
        data1 = self.fcn(data1)

        a = data1.view(-1, 2 * c, 1, 1)
        a1 = 1 - a

        y = conv_5x5 * a
        y1 = conv_3x3 * a1

        data_a_a1 = torch.add(y, y1)
        channel_data = self.conv_1x1(data_a_a1)

        # Spatial attention
        data2 = torch.add(channel_data, spatil_data)
        b = self.relu(data2)
        b1 = 1 - b

        z = b * channel_data
        z1 = b1 * spatil_data
        data_b_b1 = torch.add(z, z1)

        return self.conv(data_b_b1)


class AAU_Net(nn.Module):
    def __init__(self, in_channels, classes_num, channels_list):
        super(AAU_Net, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        channels = in_channels

        # 编码器部分
        for out_channels in channels_list:
            if out_channels > 32:
                self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(HAAM(channels, out_channels))
            self.encoder.append(HAAM(out_channels, out_channels))
            channels = out_channels

        channels = channels_list[-1]
        # 解码器部分
        for i in range(len(channels_list) - 2, -1, -1):
            self.decoder.append(nn.ConvTranspose2d(channels, channels_list[i], kernel_size=2, stride=2))
            self.decoder.append(HAAM(2 * channels_list[i], channels_list[i]))
            self.decoder.append(HAAM(channels_list[i], channels_list[i]))
            channels = channels_list[i]

        self.out = nn.Sequential(nn.Conv2d(channels, classes_num, kernel_size=1),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        encoder_outputs = []

        # 编码器部分
        for i in range(0, len(self.encoder), 3):
            x = self.encoder[i](x)
            x = self.encoder[i + 1](x)
            encoder_outputs.append(x)
            if i < 12:
                x = self.encoder[i + 2](x)

        x = encoder_outputs[-1]

        # 解码器部分
        for i in range(0, len(self.decoder), 3):
            x = self.decoder[i](x)
            x = torch.cat([x, encoder_outputs[-2 - i // 3]], dim=1)
            x = self.decoder[i + 1](x)
            x = self.decoder[i + 2](x)

        x = self.out(x)
        return x


if __name__ == "__main__":
    input_image = torch.randn(8, 1, 224, 224)  # 输入图像尺寸为 256x256，并且有 3 个通道
    # channels_list = [32, 64, 128, 256, 512]  # 每一层的通道数列表
    model = AAU_Net(in_channels=1, classes_num=2, channels_list=[32, 64, 128, 256, 512])

    output_image = model(input_image)
    print(output_image)
    print("输出图像尺寸:", output_image.shape)
