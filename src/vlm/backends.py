"""VLM backends. Anything with `.name` and `.generate(image_rgb, prompt)` works."""
from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np


class VLMBackend(Protocol):
    name: str

    def generate(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 48) -> str: ...


class MockVLM:
    """Scripted backend for tests: `responder(image, prompt) -> text`. Records every call."""

    def __init__(self, responder: Callable[[np.ndarray, str], str], name: str = "mock-vlm"):
        self.name = name
        self.responder = responder
        self.calls: List[Tuple[Tuple[int, int], str]] = []

    def generate(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 48) -> str:
        self.calls.append((image_rgb.shape[:2], prompt))
        return self.responder(image_rgb, prompt)


class Qwen2VLBackend:
    """Qwen2-VL through Hugging Face transformers (needs transformers >= 4.45).

    The model is loaded on first use. Greedy decoding, so the same crop always
    gets the same answer. bf16 weights of the 7B model need about 16 GB of VRAM;
    `load_in_4bit` (bitsandbytes) cuts that to roughly a third.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda",
                 dtype: str = "bfloat16", max_pixels: int = 448 * 448, load_in_4bit: bool = False):
        self.name = model_id
        self.device, self.dtype, self.max_pixels, self.load_in_4bit = device, dtype, max_pixels, load_in_4bit
        self.model = None
        self.processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = getattr(torch, self.dtype) if hasattr(torch, self.dtype) else torch.bfloat16
        kw = dict(torch_dtype=dtype, attn_implementation="sdpa")

        if self.device == "cpu":
            kw["device_map"] = "cpu"
        else:
            kw["device_map"] = "auto"

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                import warnings
                warnings.warn(f"bitsandbytes quantization setup failed: {e}. Falling back to standard device_map='auto'.")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.name, **kw)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.name, max_pixels=self.max_pixels)

    def generate(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 48) -> str:
        import torch
        from PIL import Image

        if self.model is None:
            self._load()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target_device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.processor(text=[text], images=[Image.fromarray(image_rgb)], return_tensors="pt").to(target_device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
