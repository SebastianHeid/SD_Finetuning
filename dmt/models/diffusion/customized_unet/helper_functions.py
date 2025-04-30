import json

from customed_classes import (
    Down2Res1,
    Down3Res1,
    IdentityBlock,
    IdentityBlock0Up,
    IdentityBlockAttn,
    IdentityBlockUp1Res2,
    IdentityBlockUp2Res0,
    IdentityBlockUp2Res1,
    IdentityBlockUp2Res2,
    IdentityBlockUp3Res,
    IdentityBlockUp3Res0,
)
from customed_down_block_classes import CustomAttnDownBlock2D, CustomDownBlock2D
from customed_mid_block_classes import CustomUNetMidBlock2DCrossAttn
from customed_up_block_classes import CustomCrossAttnUpBlock2D, CustomUpBlock2D
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)


def get_configs(path: str):
    with open(path, "r") as f:
        configs = json.load(f)
    return configs


# def down_set_weights(
#     unet, remove_downsample_blocks, remove_resnet_blocks, att_block_trainable
# ):
#     unet.requires_grad_(False)

#     blocks = []

#     for i in range(len(remove_downsample_blocks)):
#         down_block_num = remove_downsample_blocks[i]
#         res_block_num = remove_resnet_blocks[i]

#         assert not (
#             down_block_num == 0 and res_block_num == 0
#         ), "Do not remove in the first block the first layer"
#         if res_block_num == 0:
#             down_block_num -= 1
#             res_block_num = 1
#         elif res_block_num == 1:
#             res_block_num = 0

#         blocks.append(unet.down_blocks[down_block_num].resnets[res_block_num])
#         if (
#             att_block_trainable and down_block_num != 3
#         ):  # the fourht down block does not contain attention layers
#             blocks.append(unet.down_blocks[down_block_num].attentions[res_block_num])

#     for block in blocks:
#         for param in block.parameters():
#             param.requires_grad = True


def down_set_weights(unet, res_stage, res_block, att_stage, att_block):
    unet.requires_grad_(False)
    blocks = []
    for i in range(len(res_stage)):
        stage = res_stage[i]
        block = res_block[i]
        blocks.append(unet.down_blocks[stage].resnets[block])
    for j in range(len(att_stage)):
        stage = att_stage[j]
        block = att_block[j]
        blocks.append(unet.down_blocks[stage].attentions[block])

    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def mid_set_weights(unet, res_block, att_block):
    blocks = []
    for i in range(len(res_block)):
        block = res_block[i]
        blocks.append(unet.mid_block.resnets[block])
    for j in range(len(att_block)):
        block = att_block[j]
        blocks.append(unet.mid_block.attentions[block])

    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def up_set_weights(unet, res_stage, res_block, att_stage, att_block):
    blocks = []
    for i in range(len(res_stage)):
        stage = res_stage[i]
        block = res_block[i]
        blocks.append(unet.up_blocks[stage].resnets[block])
    for j in range(len(att_stage)):
        stage = att_stage[j]
        block = att_block[j]
        blocks.append(unet.up_blocks[stage].attentions[block])

    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def replace_downblocks_in_unet(unet, config_json):
    """Replace all CrossAttnDownBlock2D blocks with custom versions"""
    print("Replacing down blocks in UNet")
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


def replace_midblocks_in_unet(unet, config_json):
    """Replace all CrossAttnDownBlock2D blocks with custom versions"""
    print("Replacing mid blocks in UNet")
    config = get_configs(config_json)

    if isinstance(unet.mid_block, UNetMidBlock2DCrossAttn):
        custom_block = CustomUNetMidBlock2DCrossAttn(**config)

        # Copy all weights
        custom_block.load_state_dict(unet.mid_block.state_dict())

        # Replace in UNet
        unet.mid_block = custom_block


def replace_upblocks_in_unet(unet, config_json):
    """Replace all CrossAttnDownBlock2D blocks with custom versions"""
    print("Replacing up blocks in UNet")
    configs = get_configs(config_json)

    for i, up_block in enumerate(unet.up_blocks):
        config = configs[i]
        if isinstance(up_block, UpBlock2D):
            custom_block = CustomUpBlock2D(**config)

            # Copy all weights
            custom_block.load_state_dict(up_block.state_dict())

            # Replace in UNet
            unet.up_blocks[i] = custom_block

        if isinstance(up_block, CrossAttnUpBlock2D):
            custom_block = CustomCrossAttnUpBlock2D(**config)

            # Copy all weights
            custom_block.load_state_dict(up_block.state_dict())

            # Replace in UNet
            unet.up_blocks[i] = custom_block


def remove_down_blocks(
    unet, resnet_stage=[], resnet_blocks=[], attn_stage=[], attn_block=[]
):
    assert len(resnet_stage) == len(
        resnet_blocks
    ), "Number of downsample_blocks and resnet_blocks must be equal"
    for i in range(len(resnet_stage)):
        down_block_num = resnet_stage[i]
        res_block_num = resnet_blocks[i]
        if down_block_num == 1 and res_block_num == 0:
            place_holder_module = Down2Res1()
        elif down_block_num == 2 and res_block_num == 0:
            place_holder_module = Down3Res1()
        else:
            place_holder_module = IdentityBlock()
        unet.down_blocks[down_block_num].resnets[res_block_num] = place_holder_module
    for i in range(len(attn_stage)):
        down_block_num = attn_stage[i]
        attn_block_num = attn_block[i]
        place_holder_module = IdentityBlockAttn()
        unet.down_blocks[down_block_num].attentions[
            attn_block_num
        ] = place_holder_module


def remove_mid_blocks(unet, resnet_blocks=[], attn_block=[]):
    for i in range(len(resnet_blocks)):
        res_block_num = resnet_blocks[i]
        unet.mid_block.resnets[res_block_num] = IdentityBlock()
    for i in range(len(attn_block)):
        att_block_num = attn_block[i]
        unet.mid_block.resnets[att_block_num] = IdentityBlock()


def remove_up_blocks(
    unet, resnet_stage=[], resnet_blocks=[], attn_stage=[], attn_block=[]
):
    assert len(resnet_stage) == len(
        resnet_blocks
    ), "Number of downsample_blocks and resnet_blocks must be equal"
    for i in range(len(resnet_stage)):
        up_block_num = resnet_stage[i]
        res_block_num = resnet_blocks[i]
        if up_block_num == 1 or up_block_num == 0:
            if res_block_num == 2 and up_block_num == 1:
                place_holder_module = IdentityBlockUp1Res2()
            else:
                place_holder_module = IdentityBlock0Up()
        elif up_block_num == 2:
            if res_block_num == 0:
                place_holder_module = IdentityBlockUp2Res0()
            elif res_block_num == 1:
                place_holder_module = IdentityBlockUp2Res1()
            else:
                place_holder_module = IdentityBlockUp2Res2()
        elif up_block_num == 3:
            if res_block_num == 0:
                place_holder_module = IdentityBlockUp3Res0()
            else:
                place_holder_module = IdentityBlockUp3Res()

        unet.up_blocks[up_block_num].resnets[res_block_num] = place_holder_module

    for i in range(len(attn_stage)):
        up_block_num = attn_stage[i]
        attn_block_num = attn_block[i]
        place_holder_module = IdentityBlockAttn()
        unet.up_blocks[up_block_num].attentions[attn_block_num] = place_holder_module
