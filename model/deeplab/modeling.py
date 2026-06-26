import torch
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, convert_to_separable_conv
from .resnet import resnet101


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet101(pretrained=pretrained_backbone,
                         replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


if __name__ == '__main__':
    model = deeplabv3plus_resnet101(num_classes=1, output_stride=8)
    # convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    inputs = torch.ones(1, 3, 256, 256)
    out = model(inputs)
    print("wnet have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    print(out.shape)
