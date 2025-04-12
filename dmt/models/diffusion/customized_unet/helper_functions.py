import json

from customed_classes import (
    CustomAttnDownBlock2D,
    CustomDownBlock2D,
    Down2Res1,
    Down3Res1,
    IdentityBlock,
)
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D


def get_configs(path: str):
    with open(path, "r") as f:
        configs = json.load(f)
    return configs


def resnets_set_weights(unet, remove_downsample_blocks, remove_resnet_blocks):
    unet.requires_grad_(False)

    blocks = []

    for i in range(len(remove_downsample_blocks)):
        down_idx = remove_downsample_blocks[i]
        res_idx = remove_resnet_blocks[i]

        assert not (
            down_idx == 0 and res_idx == 0
        ), "Do not remove in the first block the first layer"
        if res_idx == 0:
            down_idx -= 1
            res_idx = 1
        elif res_idx == 1:
            res_idx = 0

        blocks.append(unet.down_blocks[down_idx].resnets[res_idx])

    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def replace_blocks_in_unet(unet, config_json):
    """Replace all CrossAttnDownBlock2D blocks with custom versions"""
    print("Replacing blocks in UNet")
    configs = get_configs(config_json)

    for i, down_block in enumerate(unet.down_blocks):
        config = configs[i]
        if isinstance(down_block, CrossAttnDownBlock2D):
            custom_block = CustomAttnDownBlock2D(**config)

            # Copy all weights
            custom_block.load_state_dict(down_block.state_dict())

            # Replace in UNet
            unet.down_blocks[i] = custom_block

        if isinstance(down_block, DownBlock2D):
            custom_block = CustomDownBlock2D(**config)

            # Copy all weights
            custom_block.load_state_dict(down_block.state_dict())

            # Replace in UNet
            unet.down_blocks[i] = custom_block


def remove_resnet_layers(unet, remove_downsample_blocks, remove_resnet_blocks):
    assert len(remove_downsample_blocks) == len(
        remove_resnet_blocks
    ), "Number of downsample_blocks and resnet_blocks must be equal"

    for i in range(len(remove_downsample_blocks)):
        down_block_num = remove_downsample_blocks[i]
        res_block_num = remove_resnet_blocks[i]
        if down_block_num == 1 and res_block_num == 0:
            place_holder_module = Down2Res1()
        elif down_block_num == 2 and res_block_num == 0:
            place_holder_module = Down3Res1()
        else:
            place_holder_module = IdentityBlock()
        unet.down_blocks[down_block_num].resnets[res_block_num] = place_holder_module
