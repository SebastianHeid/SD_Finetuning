from typing import Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


def downsample(
    self,
    emb: torch.Tensor,
    sample: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    is_multiscale=False,
):
    feat_list = []
    down_block_res_samples = (sample,)
    res_outputs = ()
    for blk_ind, downsample_block in enumerate(self.down_blocks):
        # print(blk_ind + 1, ". Downsample Block")
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}

            sample, res_samples, res_out = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        down_block_res_samples += res_samples
        res_outputs += (res_out,)
        if is_multiscale and blk_ind <= 1:
            feat_list.append(sample)
    # return sample, down_block_res_samples, res_outputs
    return sample, down_block_res_samples


def upsample(
    self,
    emb: torch.Tensor,
    sample: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    down_block_res_samples: Tuple[torch.Tensor, ...],
    upsample_size: Optional[Tuple[int, int]] = None,
):
    for i, upsample_block in enumerate(self.up_blocks):

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if (
            hasattr(upsample_block, "has_cross_attention")
            and upsample_block.has_cross_attention
        ):
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=None,
                upsample_size=upsample_size,
                attention_mask=None,
                encoder_attention_mask=None,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )
    return sample


def unet_forward(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
):

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, None)
    aug_emb = None

    aug_emb = self.get_aug_embed(
        emb=emb,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    )

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 3. down
    # # sample, down_block_res_samples, res_outputs = downsample(
    # #     self, emb, sample, encoder_hidden_states
    # # )
    sample, down_block_res_samples = downsample(
        self, emb, sample, encoder_hidden_states
    )

    # 4. mid
    if self.mid_block is not None:
        if (
            hasattr(self.mid_block, "has_cross_attention")
            and self.mid_block.has_cross_attention
        ):
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
            )
        else:
            sample = self.mid_block(sample, emb)

    # 5. up
    # print(len(down_block_res_samples))
    sample = upsample(self, emb, sample, encoder_hidden_states, down_block_res_samples)
    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return UNet2DConditionOutput(sample=sample)
