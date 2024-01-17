from typing import Optional
import gin

from segmentation.utils import MSC
from deeplab_pytorch.libs.models.deeplabv2 import DeepLabV2


def torchvision_resnet_weight_key_to_deeplab2(key: str) -> Optional[str]:
    segments = key.split('.')

    if segments[0].startswith('layer'):
        layer_num = int(segments[0].split('layer')[-1])
        dl_layer_num = layer_num + 1

        block_num = int(segments[1])
        dl_block_str = f'block{block_num + 1}'

        layer_type = segments[2]
        if layer_type == 'downsample':
            shortcut_module_num = int(segments[3])
            if shortcut_module_num == 0:
                module_name = 'conv'
            elif shortcut_module_num == 1:
                module_name = 'bn'
            else:
                raise ValueError(shortcut_module_num)

            return f'layer{dl_layer_num}.{dl_block_str}.shortcut.{module_name}.{segments[-1]}'

        else:
            layer_type, conv_num = segments[2][:-1], segments[2][-1]
            conv_num = int(conv_num)

            if conv_num == 1:
                dl_conv_name = 'reduce'
            elif conv_num == 2:
                dl_conv_name = 'conv3x3'
            elif conv_num == 3:
                dl_conv_name = 'increase'
            else:
                raise ValueError(conv_num)

            return f'layer{dl_layer_num}.{dl_block_str}.{dl_conv_name}.{layer_type}.{segments[-1]}'

    elif segments[0] in {'conv1', 'bn1'}:
        layer_type = segments[0][:-1]
        return f'layer1.conv1.{layer_type}.{segments[-1]}'

    return None


@gin.configurable(allowlist=['deeplab_n_features', 'scales'])
def deeplabv2_resnet101_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED,
                                 scales=[1.0], **kwargs):
    return MSC(
        base=DeepLabV2(
            n_classes=deeplab_n_features, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=scales,
    )
