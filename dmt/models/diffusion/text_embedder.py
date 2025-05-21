import torch as th 
import torch.nn as nn 
from transformers import CLIPTextModel, CLIPTokenizer

class TextEmbedderWrapper(nn.Module):
    def __init__(self, text_encoder, text_tokenizer):
        super().__init__()
        self.text_encoder = text_encoder 
        self.text_tokenizer = text_tokenizer 
    def forward(self, prompt):
        text_inputs = self.text_tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids
        text_inputs = text_inputs.to(self.text_encoder.device)
        with th.no_grad():
            text_emb = self.text_encoder(text_inputs)[0]
        return text_emb
        

def init_text_embedder(path_to_network: str):
    text_tokenizer = CLIPTokenizer.from_pretrained(path_to_network, subfolder="tokenizer", use_fast=True)
    text_encoder = CLIPTextModel.from_pretrained(path_to_network, subfolder="text_encoder")
    return TextEmbedderWrapper(text_encoder, text_tokenizer)