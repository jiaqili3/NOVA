# Training Guide
This guide provides simple snippets to train diffnext models.

# 1. Build VAE cache
To optimize training workflow, we preprocess images or videos into VAE latents.

## Requirements:
```bash
pip install codewithgpu decord
```

## Build T2I cache
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

## Build T2V cache
Following snippet can be used to cache video latents:

```python
import os, codewithgpu, torch, decord, numpy as np
from diffnext.models.autoencoders.autoencoder_kl_opensora import AutoencoderKLOpenSora

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderKLOpenSora.from_pretrained("/path/to/nova-d48w1024-osp480/vae")
vae = vae.to(device=device, dtype=dtype).eval()

features = {"moments": "bytes", "caption": "string", "text": "string", "shape": ["int64"], "flow": "float64"}
_, writer = os.makedirs("./vid_dataset"), codewithgpu.RecordWriter("./vid_dataset", features)

resize, crop_size, frame_ids = 480, (480, 768), list(range(0, 65, 2))
vid = decord.VideoReader("./assets/sample_video.mp4")
h, w = vid[0].shape[:2]
scale = float(resize) / float(min(h, w))
size = int(h * scale + 0.5), int(w * scale + 0.5)
y, x = (size[0] - crop_size[0]) // 2, (size[1] - crop_size[1]) // 2
vid = decord.VideoReader("./assets/sample_video.mp4", height=size[0], width=size[1])
vid = vid.get_batch(frame_ids).asnumpy()
vid = vid[:, y : y + crop_size[0], x : x + crop_size[1]]
x = torch.as_tensor(vid[None, ...].transpose((0, 4, 1, 2, 3))).to(device).to(dtype)
with torch.no_grad():
    x = vae.encode(x.sub(127.5).div(127.5)).latent_dist.parameters.cpu().numpy()[0]
example = {"caption": "long caption", "text": "short text", "flow": 5}
writer.write({"shape": x.shape, "moments": x.tobytes(), **example}), writer.close()
```

# 2. Generate model config
To simplify arguments parsing, we use [YACS](https://github.com/rbgirshick/yacs) to enhance ``diffusers``.

## Generate T2I config
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

## Generate T2V config
Following snippet provides simple T2V training arguments:

```python
from diffnext.config import cfg
cfg.NUM_GPUS = 1
cfg.PIPELINE.TYPE = "nova_train_t2v"
cfg.MODEL.PRECISION = "bfloat16"
cfg.MODEL.WEIGHTS = "/path/to/nova-d48w1024-osp480"
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.MAX_STEPS = 100
cfg.SOLVER.EMA_EVERY = 100
cfg.SOLVER.SNAPSHOT_EVERY = 100
cfg.SOLVER.SNAPSHOT_PREFIX = "nova_d48w1024_480px"
cfg.TRAIN.DATASET = "./vid_dataset"
cfg.TRAIN.LOADER = "vae_train"
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.CHECKPOINTING = 3  # 0,1,2,3
cfg.TRAIN.MODEL_EMA = 0.98
cfg.TRAIN.DEVICE_EMA = "cpu"  # "cpu", "cuda"
open("./nova_d48w1024_480px.yml", "w").write(str(cfg))
```

# 3. Run model training
To simplify distributed training, we dispatch torch/deepspeed/simple launchers into ``train.py`` script.

## Train T2I model
```bash
python -u scripts/train.py --cfg ./nova_d48w1024_1024px.yml
```

## Train T2V model
```bash
python -u scripts/train.py --cfg ./nova_d48w1024_480px.yml
```

## Dispatch deepspeed launcher
```python
deepspeed_config = {
  "train_micro_batch_size_per_gpu": 1,
  "bf16": {
    "enabled": True
  },
  "zero_optimization": {
    "stage": 2
  }
}
json.dump(open("./zero2_bf16.json", "w"), deepspeed_config)
```

```bash
python -u scripts/train.py --cfg ./nova_d48w1024_1024px.yml --deepspeed ./zero2_bf16.json
```


