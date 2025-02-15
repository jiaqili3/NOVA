# Training Guide

This guide provides step-by-step instructions on how to prepare and train the 'nova-d48w768-sdxl1024' model. The steps include downloading necessary weights, caching VAE latents, and executing the training process.


# 1. Download VAE weight

The Variational Autoencoder (VAE) weight is essential for encoding images into a latent space representation, which significantly accelerates the training process and reduces GPU memory usage. Please download the required VAE weight from the following link:

- [🤗 HF link](https://huggingface.co/BAAI/nova-d48w768-sdxl1024)

# 2. Caching VAE Latents

To optimize your training workflow, we preprocess images or videos into their latent representations. This preprocessing step is crucial for enhancing the efficiency of the training process.

Use the following script to cache image latents:

```bash
export PYTHONPATH=/path/to/NOVA

python scripts/cashe_img_latent.py \
--img_path /path/to/NOVA/data/raw_data/sample_image.jpg \
--caption_path /path/to/NOVA/data/raw_data/prompt_img.txt \
--latent_path /path/to/NOVA/data/latent/nova-d48w768-sdxl1024 \
--vae /path/to/Weight/nova-d48w768-sdxl1024/vae \
--vae-batch-size 1 \
--raw_data_size 1024
```

# 3. Train the Model
```bash
export PYTHONPATH=/path/to/NOVA

python -u scripts/train.py  \
--cfg /path/to/NOVA/configs/train/nova_d48w1024/nova_d48w1024_1024px.yml
```