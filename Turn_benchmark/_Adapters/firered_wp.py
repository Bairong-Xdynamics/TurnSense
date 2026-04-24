from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from .base import BaseTurnModel


class FireRedChatWP(BaseTurnModel):
    """
    FireRedChat-turn-detector: text binary {complete,incomplete} via ONNX
    """
    model_name = "FireRedChat-turn-detector"
    supports_audio = False
    supports_text = True
    supports_context = False
    supported_labels = {"complete", "incomplete"}

    def __init__(self, model_dir: str, lang: str = "zh", threshold: float = 0.5, use_gpu: bool = False):
        import onnxruntime as ort
        from modelscope import AutoTokenizer

        self.threshold = float(threshold)
        tok_dir = os.path.join(model_dir, "tokenizer")
        if not os.path.isdir(tok_dir):
            raise FileNotFoundError(f"tokenizer dir not found: {tok_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, truncation_side="left", use_fast=False)

        onnx_name = "chinese_best_model_q8.onnx" if lang == "zh" else "multilingual_best_model_q8.onnx"
        onnx_path = os.path.join(model_dir, onnx_name)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"onnx not found: {onnx_path}")

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "ROCMExecutionProvider" in avail:
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        text = (sample.get("last_text") or "").strip()
        if not text:
            return None

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="np",
            max_length=128,
        )
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype("int64"),
                "attention_mask": inputs["attention_mask"].astype("int64"),
            },
        )
        probs = self._softmax(outputs[0])
        eou_prob = float(probs[0, -1])
        return "complete" if eou_prob > self.threshold else "incomplete"
