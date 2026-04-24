


from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from .base import BaseTurnModel


class SmartTurnWP(BaseTurnModel):
    """
    Smart-Turn-v3: audio binary {complete,incomplete} via ONNX
    """
    model_name = "Smart-Turn-v3"
    supports_audio = True
    supports_text = False
    supports_context = False
    supported_labels = {"complete", "incomplete"}

    def __init__(self, onnx_path: str, prefer_gpu: bool = True):
        import onnxruntime as ort
        import torchaudio
        from transformers import WhisperFeatureExtractor
        import torch

        self._ort = ort
        self._torchaudio = torchaudio
        self._torch = torch
        self.extractor = WhisperFeatureExtractor(chunk_length=8)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        if prefer_gpu and ("CUDAExecutionProvider" in ort.get_available_providers()):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
            print("[SmartTurn] Using CUDAExecutionProvider.")
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = None
            print("[SmartTurn] Using CPUExecutionProvider.")

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )

    def _load_audio_16k_mono(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        wav, sr = self._torchaudio.load(audio_path)
        if wav.numel() == 0:
            return np.zeros((1,), dtype=np.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        if sr != target_sr:
            wav = self._torchaudio.functional.resample(wav, sr, target_sr)

        wav = wav.contiguous().to(self._torch.float32)
        return wav.cpu().numpy()

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        wav_path = sample.get("last_wav")
        if not wav_path:
            return None

        audio = self._load_audio_16k_mono(wav_path, 16000)
        max_samples = 8 * 16000
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        else:
            audio = np.pad(audio, (max_samples - len(audio), 0))

        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="np")["input_features"].astype(np.float32)
        outputs = self.session.run(None, {"input_features": inputs})
        prob = float(outputs[0][0].item())
        return "complete" if prob > 0.5 else "incomplete"
