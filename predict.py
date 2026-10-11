"""Replicate / Cog predictor for NeuTTS Nano (English)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path as CogPath

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"

# Prefer image-baked HF cache (populated by GitHub Actions before cog push).
for cache_dir in (ROOT / "hf_cache", Path("/src/hf_cache")):
    if cache_dir.exists():
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
        break

PRESET_VOICES = ["jo", "dave", "emily", "paul", "sophie", "steven"]


class Predictor(BasePredictor):
    def setup(self) -> None:
        from neutts import NeuTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            os.environ.setdefault("HF_TOKEN", token)

        self.tts = NeuTTS(
            backbone_repo="neuphonic/neutts-nano",
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device,
        )
        self.sample_rate = 24_000
        self._preset_cache: dict[str, tuple] = {}

        try:
            codes, ref_text = self._load_preset("jo")
            _ = self.tts.infer("Hello.", codes, ref_text)
        except Exception:
            pass

    def _load_preset(self, name: str):
        if name in self._preset_cache:
            return self._preset_cache[name]

        wav = SAMPLES / f"{name}.wav"
        txt = SAMPLES / f"{name}.txt"
        pt = SAMPLES / f"{name}.pt"
        if not txt.exists():
            raise ValueError(f"Unknown preset voice: {name}")

        ref_text = txt.read_text(encoding="utf-8").strip()
        if pt.exists():
            codes = torch.load(pt, map_location="cpu", weights_only=True)
        else:
            if not wav.exists():
                raise ValueError(f"Missing reference audio for voice: {name}")
            codes = self.tts.encode_reference(str(wav))
        self._preset_cache[name] = (codes, ref_text)
        return codes, ref_text

    def predict(
        self,
        text: str = Input(description="English text to synthesize."),
        voice: str = Input(
            description="Built-in reference voice. Ignored when ref_audio is provided.",
            default="jo",
            choices=PRESET_VOICES,
        ),
        ref_audio: Optional[CogPath] = Input(
            description="Optional reference clip (3–15s wav) for voice cloning.",
            default=None,
        ),
        ref_text: str = Input(
            description="Transcript of ref_audio (required when uploading a custom reference).",
            default="",
        ),
        temperature: float = Input(
            description="Sampling temperature.",
            default=1.0,
            ge=0.1,
            le=1.5,
        ),
        top_k: int = Input(
            description="Top-k sampling cutoff.",
            default=50,
            ge=1,
            le=200,
        ),
        seed: int = Input(
            description="Seed for reproducible generation (-1 = random).",
            default=-1,
        ),
    ) -> CogPath:
        text = (text or "").strip()
        if not text:
            raise ValueError("text must not be empty")

        if ref_audio is not None:
            ref_text = (ref_text or "").strip()
            if not ref_text:
                raise ValueError("ref_text is required when ref_audio is provided")
            codes = self.tts.encode_reference(str(ref_audio))
        else:
            codes, ref_text = self._load_preset(voice)

        self.tts._seed = None if seed is None or int(seed) < 0 else int(seed)
        wav = self.tts.infer(text, codes, ref_text, temperature=temperature, top_k=top_k)

        if wav is None or len(np.asarray(wav)) == 0:
            raise RuntimeError("Model returned empty audio")

        out = Path(tempfile.mkdtemp()) / "output.wav"
        sf.write(str(out), np.asarray(wav), self.sample_rate)
        return CogPath(out)
