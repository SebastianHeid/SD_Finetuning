from typing import Any, Dict, Optional, Tuple, Union

import diffusers.utils.logging as logging
import torch
from diffusers import UNet2DConditionModel
from unet_forward_path import unet_forward

model_id = "stabilityai/stable-diffusion-2-1"
logger = logging.get_logger(__name__)


class ModifiedUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        sample, down_block_res_samples, res_outputs, res_inputs, block_outputs = (
            unet_forward(
                self,
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )
        )
        return sample, res_outputs, block_outputs
