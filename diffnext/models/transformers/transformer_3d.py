# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Base 3D transformer model for video generation."""

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm


class Transformer3DModel(nn.Module):
    """
    Base 3D transformer model for video generation.
    
    This model combines several components to generate videos:
    1. Video encoder - processes video frames
    2. Image encoder/decoder - handles image processing
    3. Various embedding layers - for text, masks, positions, etc.
    4. Noise and sampling schedulers - for diffusion-based generation
    """

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        label_embed=None,
        video_pos_embed=None,
        image_pos_embed=None,
        motion_embed=None,
        noise_scheduler=None,
        sample_scheduler=None,
    ):
        """
        Initialize the 3D transformer model with various components.
        
        Args:
            video_encoder: Encoder for processing video frames
            image_encoder: Encoder for processing images
            image_decoder: Decoder for generating images
            mask_embed: Embedding layer for masks
            text_embed: Embedding layer for text prompts
            label_embed: Embedding layer for labels
            video_pos_embed: Positional embedding for video
            image_pos_embed: Positional embedding for images
            motion_embed: Embedding layer for motion information
            noise_scheduler: Scheduler for adding noise during training
            sample_scheduler: Scheduler for sampling during generation
        """
        super(Transformer3DModel, self).__init__()
        self.video_encoder = video_encoder
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.mask_embed = mask_embed
        self.text_embed = text_embed
        self.label_embed = label_embed
        self.video_pos_embed = video_pos_embed
        self.image_pos_embed = image_pos_embed
        self.motion_embed = motion_embed
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_scheduler
        self.pipeline_preprocess = lambda inputs: inputs
        self.loss_repeat = 4  # Number of times to repeat inputs for loss calculation

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar for tracking progress during generation."""
        return tqdm(iterable) if enable else iterable

    def preprocess(self, inputs: Dict):
        """
        Preprocess model inputs before processing.
        
        This method:
        1. Handles guidance scaling for conditional generation
        2. Processes text prompts if provided
        3. Processes motion flow information if provided
        4. Combines all conditioning information
        """
        # Check if classifier-free guidance is being used
        add_guidance = inputs.get("guidance_scale", 1) > 1
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        
        # Create empty input tensor if not provided
        if inputs.get("x", None) is None:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
        
        # Process text prompt if provided
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
        
        # Process motion flow information if provided
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            # Duplicate flow and fps for classifier-free guidance if needed
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
        
        # Combine all conditioning information
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """
        Calculate training losses for the model.
        
        Args:
            z: Latent representation [b*t, patch_num_image_encoder, hidden_dim]
            x: Target image [b*t, vae_c, h, w]
            video_shape: Shape of the video for temporal loss calculation
            
        Returns:
            Dictionary containing loss values
        """
        # Repeat inputs for loss calculation
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        
        # Convert image to patches
        x = self.image_encoder.patch_embed.patchify(x)
        
        # Add noise to the input for diffusion training
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        
        # Get prediction type from scheduler
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        
        # Generate prediction from the model
        model_pred = self.image_decoder(x_t, timestep, z) # [b*4, h, w]
        
        # Calculate target based on prediction type
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        
        # Calculate MSE loss
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        
        # Handle video-specific loss calculation
        if video_shape is not None:
            loss = loss.view((-1,) + video_shape).transpose(0, 1).sum((1, 2))
            i2i = loss[1:].sum().mul_(video_shape[0] / (video_shape[0] - 1))
            return {"loss_t2i": loss[0].mul(video_shape[0]), "loss_i2i": i2i}
        
        return {"loss": loss.sum()}

    @torch.no_grad()
    def denoise(self, z, x, guidance_scale=1, generator=None, pred_ids=None) -> torch.Tensor:
        """
        Run the diffusion denoising process to generate samples.
        
        This method implements classifier-free guidance for conditional generation.
        
        Args:
            z: Conditioning information
            x: Noisy input
            guidance_scale: Scale factor for classifier-free guidance
            generator: Random number generator
            pred_ids: IDs for prediction
            
        Returns:
            Denoised tensor
        """
        self.sample_scheduler._step_index = None  # Reset counter.
        
        # Iterate through timesteps in reverse order
        for t in self.sample_scheduler.timesteps:
            # Duplicate input for classifier-free guidance if needed
            x_pack = torch.cat([x] * 2) if guidance_scale > 1 else x
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            
            # Generate noise prediction
            noise_pred = self.image_decoder(x_pack, timestep, z, pred_ids)
            
            # Apply classifier-free guidance if needed
            if guidance_scale > 1:
                cond, uncond = noise_pred.chunk(2)
                noise_pred = uncond.add_(cond.sub_(uncond).mul_(guidance_scale))
            
            # Convert patches back to image
            noise_pred = self.image_encoder.patch_embed.unpatchify(noise_pred)
            
            # Update sample using the scheduler
            x = self.sample_scheduler.step(noise_pred, t, x, generator=generator).prev_sample
        
        # Return final result as patches
        return self.image_encoder.patch_embed.patchify(x)

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """
        Generate a single frame of video.
        
        This method:
        1. Applies guidance scaling
        2. Generates predictions in multiple steps
        3. Updates the frame with the generated content
        
        Args:
            states: Dictionary containing current generation state
            inputs: Dictionary containing generation parameters
        """
        # Get guidance parameters
        guidance_scale = inputs.get("guidance_scale", 1)
        min_guidance_scale = inputs.get("min_guidance_scale", guidance_scale)
        max_guidance_scale = inputs.get("max_guidance_scale", guidance_scale)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        
        # Get number of predictions to make
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        
        # Determine guidance scale based on current state
        guidance_end = max_guidance_scale if states["t"] else guidance_scale
        guidance_start = max_guidance_scale if states["t"] else min_guidance_scale
        
        # Initialize state variables
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None
        
        # Generate predictions in multiple steps
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            # Calculate guidance scale for this step
            guidance_level = (i + 1) / len(all_num_preds)
            guidance_scale = (guidance_end - guidance_start) * guidance_level + guidance_start
            
            # Process input
            z = self.mask_embed(self.image_encoder.patch_embed(x))
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = torch.cat([z] * 2) if guidance_scale > 1 else z
            
            # Encode input
            z = self.image_encoder(z, c, prev_ids, pos=pos)
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            
            # Generate noise and denoise
            states["noise"].normal_(generator=generator)
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids)
            
            # Update frame with generated content
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """
        Generate a complete video sequence.
        
        This method:
        1. Sets up the generation process
        2. Generates frames sequentially
        3. Handles temporal consistency between frames
        
        Args:
            inputs: Dictionary containing generation parameters
        """
        # Get generation parameters
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1)
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        
        # Initialize state
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        
        # Set up positional embeddings
        if self.image_pos_embed:
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length)
        
        # Enable KV cache for efficiency if generating multiple frames
        self.video_encoder.enable_kvcache(max_latent_length > 1)
        
        # Generate frames sequentially
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            # Get positional embedding for current frame
            pos = time_pos[states["t"]] if time_pos else None
            
            # Process current frame
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c
            c = torch.cat([c] * 2) if guidance_scale > 1 else c
            
            # Encode frame
            c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos)
            
            # Apply temporal mixing if available
            if not isinstance(self.video_encoder.mixer, torch.nn.Identity):
                states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
                states["*"] = states["*"] if states["t"] else states["c"]
            
            # Generate frame
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
        
        # Disable KV cache
        self.video_encoder.enable_kvcache(False)

    def train_video(self, inputs):
        """
        Train the model on a batch of videos.
        
        This method implements a three-stage training process:
        1. 3D temporal autoregressive modeling (TAM)
        2. 2D masked autoregressive modeling (MAM)
        3. 1D token-wise diffusion modeling
        
        Args:
            inputs: Dictionary containing training data
            
        Returns:
            Dictionary containing loss values
        """
        # 3D temporal autoregressive modeling (TAM)
        # Ensure input has temporal dimension
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None # x: [b, vae_c, t, h, w]
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)
        
        # Process all frames except the last one (which is the target)
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1]) # [b, t-1, num_patches, c]
        
        # Add beginning of video token
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1) # [b, 1, num_patches, c]
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None
        
        # Get positional embeddings if available
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        
        # Set attention mask for temporal consistency
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None # inputs["c"] is the text embedding
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        
        # Encode video frames to latents (condition c)
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos) # [b, t*num_patches, c=1024] inputs["c"] is the text embedding
        
        # Apply temporal mixing if available
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)
        
        # 2D masked autoregressive modeling (MAM)
        # Reshape input for image processing
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1) # [b*t, vae_c, h, w]
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length # [b*t, num_patches_image_encoder, c=1024]
        
        # Get positional embeddings if available
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        
        # predict the masked patches conditioned on the video latents
        z = self.image_encoder(self.mask_embed(z), c.reshape(bs, -1, c.size(-1)), pos=pos) # [b*t, num_patches_image_encoder, c=1024]
        
        # 1D token-wise diffusion modeling
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return self.get_losses(z, x, video_shape=video_shape)

    def forward(self, inputs):
        """
        Define the computation performed at every call.
        
        This method:
        1. Preprocesses inputs
        2. Calls appropriate method based on training mode
        3. Returns generated video frames
        
        Args:
            inputs: Dictionary containing model inputs
            
        Returns:
            Dictionary containing generated video frames
        """
        # Preprocess inputs
        self.pipeline_preprocess(inputs)
        self.preprocess(inputs)
        
        # Call appropriate method based on training mode
        if self.training:
            return self.train_video(inputs)
        
        # Generate video
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_video(inputs)
        return {"x": torch.stack(inputs["latents"], dim=2)}
