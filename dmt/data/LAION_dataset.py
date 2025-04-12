import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LAIONDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
    ):
        # Initialize only non-GPU components
        self.dataset_dir = dataset_dir
        if split == "val" or split == "test":
            self.img_names = os.listdir(dataset_dir + "/" + split + "/latents")[:100]
        else:
            self.img_names = os.listdir(dataset_dir + "/" + split + "/latents")[:100]
        self.prompts = dict()

        print("Load prompts ...")
        self.prompts_emb = {
            img_name: torch.load(
                os.path.join(dataset_dir + "/" + split + "/txt_embs/" + img_name)
            )["encoder_hidden_states"].float()
            for img_name in tqdm(self.img_names)
        }

        print("Load latents ...")
        self.latents = {
            img_name: torch.load(
                os.path.join(dataset_dir + "/" + split + "/latents/" + img_name)
            ).float()
            for img_name in tqdm(self.img_names)
        }

    def __getitem__(self, id: int):
        image_name = self.img_names[id]
        prompt_emb = self.prompts_emb[image_name]
        latent = self.latents[image_name]

        return {
            "pixel_values": latent,  # Tensor (3, h, w) in range [-1, 1]
            "prompt_emb": prompt_emb,
        }

    def __len__(self):
        return len(self.img_names)
