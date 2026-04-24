from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from .base import BaseTurnModel, ort_providers


class NAMOTurnWP(BaseTurnModel):
    """
    NAMO-Turn-Detector: text binary {complete,incomplete} via ONNX
    """
    model_name = "NAMO-Turn-Detector"
    supports_audio = False
    supports_text = True
    supports_context = False
    supported_labels = {"complete", "incomplete"}

    def __init__(self, local_dir: str, filename: str = "model_quant.onnx", prefer_gpu: bool = True, providers=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = os.path.join(local_dir, filename)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        # providers 逻辑：默认 prefer_gpu=True -> CUDA 可用则用 CUDA
        if providers is None:
            providers, provider_options = ort_providers(prefer_gpu)
        else:
            provider_options = None

        # ---------- 加载 tokenizer（fast -> slow 自动 fallback） ----------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                local_files_only=True,
                use_fast=True
            )
        except Exception as e_fast:
            print("[NAMO] [WARN] Fast tokenizer load failed, fallback to slow tokenizer.")
            print(f"[NAMO] [WARN] fast tokenizer error: {e_fast}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                local_files_only=True,
                use_fast=False
            )

        # ---------- 加载 ONNX ----------
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
            provider_options=provider_options,
        )

        # 打印确认：是否真的用上 CUDA provider
        try:
            print("[NAMO] ORT providers:", self.session.get_providers())
        except Exception:
            pass

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.max_length = int(getattr(self.tokenizer, "model_max_length", 512) or 512)

    @staticmethod
    def _softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        text = (sample.get("last_text") or "").strip()
        if not text:
            return None

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = inputs["input_ids"]
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            )

        outputs = self.session.run(None, feed)
        logits = outputs[0]  # [1, 2]
        probs = self._softmax(logits[0])
        label = int(np.argmax(probs))  # 0=Not End, 1=End
        return "complete" if label == 1 else "incomplete"
