# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Cache image latents."""

import os
import argparse
import json
import torch
import codewithgpu
import numpy as np
from torchvision import transforms
from PIL import Image

from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Build images cache.")
    parser.add_argument("--img_path", type=str, help="path of image")
    parser.add_argument("--caption_path", type=str, help="path of caption")
    parser.add_argument("--latent_path", type=str, help="path of latent")
    parser.add_argument("--vae", type=str, help="path to VAE model")
    parser.add_argument("--vae-batch-size", type=int, default=64, help="VAE inference batch size")
    parser.add_argument("--raw_data_size", type=int, default=512, help="VAE inference batch size")
    return parser.parse_args()


def cache_img_text_pair(img_path, caption_path, writer, vae, batch_size, resize=256):
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    caption = []
    with open(caption_path, "r", encoding="utf-8") as file:
        caption = file.read().strip()
    short_caption = caption.split("/")[0]

    # convert to latent
    img = Image.open(img_path).convert("RGB")
    image_transform = transform(img).to(vae.device, vae.dtype).unsqueeze(0)
    with torch.no_grad():
        latents = vae.encode(image_transform).latent_dist.parameters.cpu().numpy()

    # The data must be for a single sample when writing.
    latent = latents[0]
    example = {
        "text": short_caption,
        "caption": caption,
        "moments": latent.tobytes(),
        "shape": latent.shape,
    }
    writer.write(example)


if __name__ == "__main__":
    # 1. args
    args = parse_args()

    # 2. build vae
    device, dtype = torch.device("cuda"), torch.float16
    vae = AutoencoderKL.from_pretrained(args.vae)
    vae = vae.to(device=device, dtype=dtype).eval()

    # 3. data format
    features = {
        "text": "string",
        "caption": "string",
        "moments": "bytes",
        "shape": ["int64"],
    }

    # 4. cashe latent
    if not os.path.exists(args.latent_path):
        os.makedirs(args.latent_path)
    writer = codewithgpu.RecordWriter(args.latent_path, features, zfill_width=6)
    cache_img_text_pair(
        args.img_path,
        args.caption_path,
        writer,
        vae,
        args.vae_batch_size,
        resize=args.raw_data_size,
    )
    writer.close()
