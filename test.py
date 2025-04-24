import random

import numpy as np
import torch
import yaml
from box import Box
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from dmt.models.diffusion.customized_unet import ModifiedUNet2DConditionModel
from dmt.models.diffusion.customized_unet.helper_functions import (
    remove_resnet_layers,
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

model_id = "stabilityai/stable-diffusion-2-1"
config_json_down = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_DownBlocks.json"
config_json_up = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_UpBlock.json"
config_jso_mid = "/export/home/sheid/SD_Finetuning/dmt/models/diffusion/customized_unet/json/SD_MidBlocks.json"

remove_downsample_blocks = [1]
remove_resnet_blocks = [1]

sd_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
unet = ModifiedUNet2DConditionModel(**sd_unet.config)
unet.load_state_dict(sd_unet.state_dict())
replace_downblocks_in_unet(unet, config_json_down)
replace_midblocks_in_unet(unet, config_jso_mid)
replace_upblocks_in_unet(unet, config_json_up)
# remove_resnet_layers(unet, remove_downsample_blocks, remove_resnet_blocks)
pip = StableDiffusionPipeline.from_pretrained(model_id)
pip.unet = unet
pip.to("cuda")
img = pip("A dangerous lion.").images[0]
img.save("./lion_down_mid_up.png")
