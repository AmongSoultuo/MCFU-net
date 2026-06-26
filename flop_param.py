from thop import profile
import torch
from model.Unet import Unet
from model.Unetpp import UnetPlusPlus
from model.AAU_Net import AAU_Net
from model.our_model import our_net
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
model_name = 'R50-ViT-B_16'
config_vit = CONFIGS_ViT_seg[model_name]
config_vit.n_classes = 1
config_vit.n_skip = 3  # 这里的n_skip含义不明,R50为3,别的用0
if model_name.find('R50') != -1:
    config_vit.patches.grid = (
        int(256 / 16), int(256 / 16))
model = deeplabv3plus_resnet50(num_classes=1, output_stride=16)
randn_input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
