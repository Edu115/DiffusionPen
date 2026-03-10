"""
DiffusionPen predictor for Replicate.

This file goes in the root of a DiffusionPen fork, alongside unet.py,
feature_extractor.py, etc. It wraps the sampling pipeline as a Cog
predictor so Replicate can serve it as an API.
"""

import json
import os
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image, ImageOps
from cog import BasePredictor, Input, Path

WEIGHTS_DIR = "/src/weights"
SAVE_PATH = os.path.join(WEIGHTS_DIR, "diffusionpen_iam_model_path")
STYLE_PATH = os.path.join(WEIGHTS_DIR, "style_models", "iam_style_diffusionpen.pth")
SD_PATH = os.path.join(WEIGHTS_DIR, "stable-diffusion-v1-5")

IMG_HEIGHT = 64
IMG_WIDTH = 256


class Predictor(BasePredictor):
    def setup(self):
        """Load all models into GPU memory."""

        # unet.py and feature_extractor.py are in the same repo
        from unet import UNetModel
        from feature_extractor import ImageEncoder
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CanineModel, CanineTokenizer

        self.device = torch.device("cuda")

        # Image transform (matches training normalisation)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),
        ])

        # CANINE text encoder
        self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = text_encoder.to(self.device).eval()

        # Style classes and character set (from train.py)
        style_classes = 339
        chars = list(
            '!"#&\'()*+,-./0123456789:;?'
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
        )

        # Args namespace matching what UNetModel expects
        class _Args:
            device = self.device
            img_size = (IMG_HEIGHT, IMG_WIDTH)
            latent = True
            color = True
            img_feat = True
            model_name = "diffusionpen"
            channels = 4
            emb_dim = 320
            num_heads = 4
            num_res_blocks = 1
            dataparallel = False

        # UNet
        unet = UNetModel(
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            in_channels=4,
            model_channels=320,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=4,
            num_classes=style_classes,
            context_dim=320,
            vocab_size=len(chars),
            text_encoder=text_encoder,
            args=_Args(),
        )

        # Load EMA weights, strip DataParallel "module." prefix
        ema_path = os.path.join(SAVE_PATH, "models", "ema_ckpt.pt")
        raw = torch.load(ema_path, map_location=self.device)
        cleaned = {k.replace("module.", ""): v for k, v in raw.items()}
        unet.load_state_dict(cleaned, strict=False)
        self.model = unet.to(self.device).eval()

        # VAE from Stable Diffusion v1-5
        self.vae = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae")
        self.vae = self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # DDIM scheduler
        self.ddim = DDIMScheduler.from_pretrained(SD_PATH, subfolder="scheduler")

        # Style encoder (MobileNetV2) — used when style reference images
        # are provided. Without images, the UNet falls back to its learned
        # class embeddings from style_id, which still works.
        feat_ext = ImageEncoder(
            model_name="mobilenetv2_100",
            num_classes=0,
            pretrained=True,
            trainable=True,
        )
        sd = torch.load(STYLE_PATH, map_location=self.device)
        md = feat_ext.state_dict()
        sd = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
        md.update(sd)
        feat_ext.load_state_dict(md)
        self.feature_extractor = feat_ext.to(self.device)
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

        print("All models loaded.")

    def _prep_style_images(self, paths):
        """Load and preprocess user-provided style reference images."""
        tensors = []
        for p in paths:
            img = Image.open(str(p)).convert("RGB")
            w, h = img.size
            img = img.resize((int(w * 64 / h), 64))
            w, _ = img.size
            if w < 256:
                img = ImageOps.pad(img, (256, 64), color="white")
            elif w > 256:
                img = img.resize((256, 64))
            tensors.append(self.img_transform(img))
        # Pad to 5 if fewer provided
        while len(tensors) < 5:
            tensors.append(tensors[-1])
        return torch.stack(tensors[:5]).to(self.device)

    def _sample_word(self, word, style_id, style_feat=None):
        """Generate a single handwritten word via 50-step DDIM diffusion."""
        n = 1

        text_feat = self.tokenizer(
            word,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(self.device)

        labels = torch.tensor([style_id]).long().to(self.device)

        # Initial latent noise (4 channels, 8x downscaled)
        x = torch.randn(
            (n, 4, IMG_HEIGHT // 8, IMG_WIDTH // 8), device=self.device
        )

        # 50-step DDIM denoising loop
        self.ddim.set_timesteps(50)
        with torch.no_grad():
            for t_step in self.ddim.timesteps:
                t = (torch.ones(n) * t_step.item()).long().to(self.device)
                noise_pred = self.model(
                    x, t, text_feat, labels,
                    original_images=None,
                    mix_rate=None,
                    style_extractor=style_feat,
                )
                x = self.ddim.step(noise_pred, t_step, x).prev_sample

        # Decode latent to pixels via VAE
        latents = x / 0.18215
        with torch.no_grad():
            decoded = self.vae.decode(latents).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        arr = decoded.cpu().squeeze(0).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr).convert("L")

    def _crop_whitespace(self, img):
        """Crop white borders from a grayscale PIL image."""
        import cv2
        arr = np.array(img)
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(th)
        if coords is None:
            return img
        bx, by, bw, bh = cv2.boundingRect(coords)
        return Image.fromarray(arr[by:by + bh, bx:bx + bw])

    def predict(
        self,
        text: str = Input(
            description="Text to render as handwriting (max 500 chars)",
            default="Happy birthday!",
        ),
        style_id: int = Input(
            description="Writer style index (0-339). Each value produces "
                        "different handwriting. Try 12, 25, 129, 201.",
            default=12,
            ge=0,
            le=339,
        ),
        style_images: Path = Input(
            description="Optional: ZIP of 5 handwriting sample images "
                        "(word-level, white background) to condition the style. "
                        "If not provided, uses learned style from style_id.",
            default=None,
        ),
        max_line_width: int = Input(
            description="Max pixel width per line before wrapping",
            default=1000,
            ge=400,
            le=2000,
        ),
    ) -> Path:
        """Generate a paragraph of handwritten text, word by word."""
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 500:
            raise ValueError("text too long (max 500 chars)")

        # Handle optional style reference images
        style_feat = None
        if style_images is not None:
            import zipfile
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(str(style_images), "r") as zf:
                    zf.extractall(tmpdir)
                img_paths = sorted([
                    os.path.join(tmpdir, f)
                    for f in os.listdir(tmpdir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if img_paths:
                    style_imgs = self._prep_style_images(img_paths)
                    style_imgs = style_imgs.reshape(-1, 3, 64, 256)
                    style_feat = self.feature_extractor(style_imgs).to(self.device)
                    print(f"Using {len(img_paths)} style reference images")

        words = text.split()
        print(f"Generating {len(words)} words in style {style_id}...")

        # Generate each word
        word_images = []
        for w in words:
            print(f"  word: {w}")
            img = self._sample_word(w, style_id=style_id, style_feat=style_feat)
            img = self._crop_whitespace(img)
            word_images.append((w, img))

        # Average character width from the longest word
        longest = max(words, key=len)
        longest_img = next(im for w, im in word_images if w == longest)
        avg_cw = longest_img.width / max(len(longest), 1)

        # Composite words into lines
        height = 64
        gap_w = 16
        gap = np.ones((height, gap_w), dtype=np.uint8) * 255

        lines = []
        cur_line = gap.copy()
        cur_w = gap_w

        for word, img in word_images:
            ar = img.width / max(img.height, 1)
            sw = max(int(avg_cw * len(word)), 10)
            sh = max(int(sw / ar), 1) if ar > 0 else height
            scaled = img.resize((sw, sh))

            if word in string.punctuation:
                par = scaled.width / max(scaled.height, 1)
                psize = (
                    (max(int(5 * par), 3), 5)
                    if word == "."
                    else (max(int(13 * par), 5), 13)
                )
                scaled = scaled.resize(psize)
                pt = max(height - scaled.height - 10, 0)
                padded = np.pad(
                    np.array(scaled),
                    ((pt, 10), (0, 0)),
                    mode="constant", constant_values=255,
                )
            elif scaled.height < height:
                p = (height - scaled.height) // 2
                padded = np.pad(
                    np.array(scaled),
                    ((p, height - scaled.height - p), (0, 0)),
                    mode="constant", constant_values=255,
                )
            else:
                scaled = scaled.resize(
                    (max(int(height * ar) - 4, 10), height - 4)
                )
                p = (height - scaled.height) // 2
                padded = np.pad(
                    np.array(scaled),
                    ((p, height - scaled.height - p), (0, 0)),
                    mode="constant", constant_values=255,
                )

            block_w = padded.shape[1] + gap_w
            if cur_w + block_w > max_line_width:
                rem = max_line_width - cur_w
                if rem > 0:
                    cur_line = np.concatenate(
                        (cur_line, np.ones((height, rem), dtype=np.uint8) * 255),
                        axis=1,
                    )
                lines.append(cur_line)
                cur_line = np.concatenate((gap, padded, gap), axis=1)
                cur_w = gap_w + padded.shape[1] + gap_w
            else:
                cur_line = np.concatenate((cur_line, padded, gap), axis=1)
                cur_w += block_w

        # Flush last line
        rem = max_line_width - cur_w
        if rem > 0:
            cur_line = np.concatenate(
                (cur_line, np.ones((height, rem), dtype=np.uint8) * 255),
                axis=1,
            )
        lines.append(cur_line)

        paragraph = np.concatenate(lines, axis=0)
        result = Image.fromarray(paragraph).convert("L")

        out_path = "/tmp/handwriting.png"
        result.save(out_path, optimize=True)
        print(f"Output: {result.size[0]}x{result.size[1]}px")
        return Path(out_path)
