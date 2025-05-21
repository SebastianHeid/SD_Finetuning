import json 
from PIL import Image
import os
import random
image_path = "/export/data/vislearn/rother_subgroup/rother_datasets/LaionAE/laion2B-en-art_512/"
image_names = [f for f in os.listdir(image_path) if f.endswith(".webp") ]
with open(train_file, "w") as file:
    for idx, img_name in enumerate(image_names):
        if idx > 100000:
            file.write(img_name + "\n")
        else:
            continue

with open(val_file, "w") as file:
    for idx, img_name in enumerate(image_names):
        if idx < 50000:
            file.write(img_name + "\n")
        else:
            break

with open(test_file, "w") as file:
    for idx, img_name in enumerate(image_names):
        if idx < 100000 and idx >= 50000:
            file.write(img_name + "\n")
        else:
            continue