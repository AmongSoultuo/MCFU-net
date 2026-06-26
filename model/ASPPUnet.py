import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, groups=in_ch, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, groups=in_ch, dilation=6, padding=6, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, groups=in_ch, dilation=12, padding=12, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, groups=in_ch, dilation=18, padding=18, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return torch.cat([x1,x2,x3,x4, x], dim=1)


class ASPPUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPPUnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 64)
        self.conv2 = DoubleConv(64, 128)
        self.down2 = Down(128, 128)
        self.conv3 = DoubleConv(128, 256)
        self.down3 = Down(256, 256)
        self.conv4 = DoubleConv(256, 512)
        self.aspp = ASPP(512, 512)
        self.down4 = Down(2560, 1024)
        self.conv5 = DoubleConv(1024, 1024)
        self.up6 = Up(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = Up(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = Up(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = Up(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.down1(c1)
        c2 = self.conv2(p1)
        p2 = self.down2(c2)
        c3 = self.conv3(p2)
        p3 = self.down3(c3)
        c4 = self.conv4(p3)
        c4_1 = self.aspp(c4)
        p4 = self.down4(c4_1)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.ones(1, 3, 256, 256).to(device)
    model = ASPPUnet(3, 1).to(device)
    # out = model(inputs)
    # print(out.shape)
    # summary(model, (3, 256, 256))
    print("wnet have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))