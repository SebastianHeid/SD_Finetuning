import random

import numpy as np
import torch
import yaml
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from dmt.models.diffusion.customized_unet import ModifiedUNet2DConditionModel
from dmt.models.diffusion.customized_unet.helper_functions import (
    remove_down_blocks,
    remove_mid_blocks,
    remove_up_blocks,
    replace_downblocks_in_unet,
    replace_midblocks_in_unet,
    replace_upblocks_in_unet,
)

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_id = "stabilityai/stable-diffusion-2-1-base"
config_json_down = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_DownBlocks.json"
config_json_up = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_UpBlock.json"
config_jso_mid = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_MidBlocks.json"

remove_downsample_blocks = []  # 0-3
remove_resnet_blocks = []  # 0-1
attn_stage = []  # 0-2
attn_block = []  # 0-1

remove_midsample_blocks = [0]  # 0-3
remove_mid_resnet_blocks = [1]  # 0-1
attn_stage_mid = [0]  # 0-2
attn_block_mid = [0]  # 0-1

remove_upsample_blocks = [3]  # 0-3
remove_up_resnet_blocks = [2]  # 0-2
attn_stage_up = [1]  # 1-3
attn_block_up = [0]  # 0-2
sd_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
unet = ModifiedUNet2DConditionModel(**sd_unet.config)
unet.load_state_dict(sd_unet.state_dict())
replace_downblocks_in_unet(unet, config_json_down)
replace_midblocks_in_unet(unet, config_jso_mid)
replace_upblocks_in_unet(unet, config_json_up)
# remove_down_blocks(
#     unet, remove_downsample_blocks, remove_resnet_blocks, attn_stage, attn_block
# )
remove_mid_blocks(
    unet,
    remove_midsample_blocks,
    remove_mid_resnet_blocks,
    attn_stage_mid,
    attn_block_mid,
)
# remove_up_blocks(unet, remove_upsample_blocks, remove_up_resnet_blocks)
pip = StableDiffusionPipeline.from_pretrained(model_id)
pip.unet = unet
pip.to("cuda")
print(sum(p.numel() for p in unet.parameters() if p.requires_grad))
img = pip("A dangerous lion.").images[0]

img.save("./lion_down_mid_up.png")
