# Training Guide

This guide provides simple snippets to train diffnext models.

# 1. Cache VAE latents

To optimize training workflow, we preprocess images or videos into VAE latents.

Following snippet can be used to cache image latents:

```python
import os, codewithgpu, torch, PIL.Image, numpy as np
from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderKL.from_pretrained("/path/to/nova-d48w1024-sdxl1024/vae")
vae = vae.to(device=device, dtype=dtype).eval()

features = {"moments": "bytes", "caption": "string", "text": "string", "shape": ["int64"]}
_, writer = os.makedirs("./img_dataset"), codewithgpu.RecordWriter("./img_dataset", features)

img = PIL.Image.open("./assets/sample_image.jpg")
x = torch.as_tensor(np.array(img)[None, ...].transpose(0, 3, 1, 2)).to(device).to(dtype)
with torch.no_grad():
    x = vae.encode(x.sub(127.5).div(127.5)).latent_dist.parameters.cpu().numpy()[0]
example = {"caption": "long caption", "text": "short text"}
writer.write({"shape": x.shape, "moments": x.tobytes(), **example}), writer.close()
```

# 2. Generate model config

To simplify arguments parsing, we use [YACS](https://github.com/rbgirshick/yacs) to enhance ``diffusers``.

Following snippet provides simple T2I training arguments:

```python
from diffnext.config import cfg
cfg.NUM_GPUS = 1
cfg.PIPELINE.TYPE = "nova_train_t2i"
cfg.MODEL.PRECISION = "bfloat16"
cfg.MODEL.WEIGHTS = "/path/to/nova-d48w1024-sdxl1024"
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.MAX_STEPS = 100
cfg.SOLVER.EMA_EVERY = 100
cfg.SOLVER.SNAPSHOT_EVERY = 100
cfg.SOLVER.SNAPSHOT_PREFIX = "nova_d48w1024_1024px"
cfg.TRAIN.DATASET = "./img_dataset"
cfg.TRAIN.LOADER = "vae_train"
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.CHECKPOINTING = 3  # 0,1,2,3
cfg.TRAIN.MODEL_EMA = 0.98
cfg.TRAIN.DEVICE_EMA = "cpu"  # "cpu", "cuda"
open("./nova_d48w1024_1024px.yml", "w").write(str(cfg))
```

# 3. Train model

```bash
python -u scripts/train.py --cfg ./nova_d48w1024_1024px.yml
```
