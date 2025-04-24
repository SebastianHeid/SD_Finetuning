import os

import torch
from diffusers import UNet2DConditionModel
from torch.utils.data import Dataset
from tqdm import tqdm


class LAIONDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        # Initialize only non-GPU components
        self.dataset_dir = dataset_dir
        if split == "val" or split == "test":
            self.img_names = os.listdir(dataset_dir + "/" + split + "/latents")[:100]
        else:
            self.img_names = os.listdir(dataset_dir + "/" + split + "/latents")
        # self.prompts = dict()

        # print("Load prompts ...")
        # self.prompts_emb = {
        #     img_name: torch.load(
        #         os.path.join(dataset_dir + "/" + split + "/txt_embs/" + img_name)
        #     )["encoder_hidden_states"]
        #     .float()
        #     .detach()
        #     for img_name in tqdm(self.img_names)
        # }

        # print("Load latents ...")
        # self.latents = {
        #     img_name: torch.load(
        #         os.path.join(dataset_dir + "/" + split + "/latents/" + img_name)
        #     )
        #     .float()
        #     .detach()
        #     for img_name in tqdm(self.img_names)
        # }

    def __getitem__(self, id: int):
        img_name = self.img_names[id]
        prompt_emb = torch.load(
            os.path.join(self.dataset_dir + "/" + self.split + "/txt_embs/" + img_name)
        )["encoder_hidden_states"].detach()
        latent = (
            torch.load(
                os.path.join(
                    self.dataset_dir + "/" + self.split + "/latents/" + img_name
                )
            )
            .float()
            .detach()
        )
        return {
            "pixel_values": latent,  # Tensor (3, h, w) in range [-1, 1]
            "prompt_emb": prompt_emb,
        }

    def __len__(self):
        return len(self.img_names)


# import os
# import torch
# from torch.utils.data import Dataset
# from collections import OrderedDict
# import gc

# class LAIONDataset(Dataset):
#     def __init__(
#         self,
#         dataset_dir: str,
#         split: str,
#         cache_size: int = 100,  # Number of items to keep in cache
#         device: str = "cuda"
#     ):
#         """
#         Memory-efficient LAION dataset loader

#         Args:
#             dataset_dir: Path to the dataset directory
#             split: Dataset split ('train', 'val', 'test')
#             cache_size: Maximum number of items to keep in memory cache
#             device: Device to use for computation
#         """
#         self.dataset_dir = dataset_dir
#         self.split = split
#         self.device = device
#         self.cache_size = cache_size

#         # Get list of image names
#         split_dir = os.path.join(dataset_dir, split)
#         latents_dir = os.path.join(split_dir, "latents")
#         if split == "val" or split == "test":
#             self.img_names = os.listdir(latents_dir)[:100]
#         else:
#             self.img_names = os.listdir(latents_dir)

#         # Define paths for loading data
#         self.latents_path = os.path.join(split_dir, "latents")
#         self.embeddings_path = os.path.join(split_dir, "txt_embs")

#         # Initialize LRU caches with OrderedDict
#         self._latent_cache = OrderedDict()
#         self._embedding_cache = OrderedDict()

#         print(f"Found {len(self.img_names)} images in {split} split")

#     def _get_latent(self, img_name):
#         """Get latent with caching"""
#         if img_name in self._latent_cache:
#             # Move to the end of OrderedDict (most recently used)
#             latent = self._latent_cache.pop(img_name)
#             self._latent_cache[img_name] = latent
#             return latent

#         # Load from disk
#         latent_path = os.path.join(self.latents_path, img_name)
#         latent = torch.load(latent_path).float().detach()

#         # Add to cache
#         if len(self._latent_cache) >= self.cache_size:
#             # Remove oldest item
#             self._latent_cache.popitem(last=False)
#         self._latent_cache[img_name] = latent

#         return latent

#     def _get_embedding(self, img_name):
#         """Get text embedding with caching"""
#         if img_name in self._embedding_cache:
#             # Move to the end of OrderedDict (most recently used)
#             embedding = self._embedding_cache.pop(img_name)
#             self._embedding_cache[img_name] = embedding
#             return embedding

#         # Load from disk
#         embedding_path = os.path.join(self.embeddings_path, img_name)
#         embedding = torch.load(embedding_path)["encoder_hidden_states"].float().detach()

#         # Add to cache
#         if len(self._embedding_cache) >= self.cache_size:
#             # Remove oldest item
#             self._embedding_cache.popitem(last=False)
#         self._embedding_cache[img_name] = embedding

#         return embedding

#     def __getitem__(self, idx: int):
#         # Get image name
#         image_name = self.img_names[idx]

#         # Load data with caching
#         latent = self._get_latent(image_name)
#         prompt_emb = self._get_embedding(image_name)

#         # Move to the target device if needed
#         if self.device != "cpu":
#             latent = latent.to(self.device)
#             prompt_emb = prompt_emb.to(self.device)

#         return {
#             "pixel_values": latent,
#             "prompt_emb": prompt_emb,
#         }

#     def __len__(self):
#         return len(self.img_names)

#     def clear_cache(self):
#         """Clear the cache to free memory"""
#         self._latent_cache.clear()
#         self._embedding_cache.clear()
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def prefetch(self, indices):
#         """Prefetch a batch of data into cache"""
#         for idx in indices:
#             if idx >= len(self):
#                 continue
#             image_name = self.img_names[idx]
#             self._get_latent(image_name)
#             self._get_embedding(image_name)
