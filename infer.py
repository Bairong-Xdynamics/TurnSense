import os
from typing import List, Dict, Any, Optional

import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

from frontend import AudioFrontend


TASK_NAME = "CompleteIncompleteInvalid"
LABELS = ["complete", "incomplete", "invalid"]

DEFAULT_ONNX_PATH = "models/model_fp32.onnx"
DEFAULT_INPUT_WAV = "/TurnSense/examples/complete.wav"

DEFAULT_AUDIO_SECONDS = 8
MAX_AUDIO_SECONDS = 8
DEFAULT_CLIP_MODE = "tail"
SAMPLING_RATE = 16000

DEFAULT_FRONTEND_CONF = {
    "cmvn_file": "models/am.mvn",
    "fs": 16000,
    "window": "hamming",
    "n_mels": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "lfr_m": 7,
    "lfr_n": 6,
    "dither": 0.0,
}


def truncate_audio(audio_array: np.ndarray, n_seconds: int, sample_rate: int, clip_mode: str) -> np.ndarray:
    max_samples = n_seconds * sample_rate
    if len(audio_array) <= max_samples:
        return audio_array

    if clip_mode == "head":
        return audio_array[:max_samples]
    return audio_array[-max_samples:]


def normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        if audio.shape[0] <= 8:
            audio = np.mean(audio, axis=0)
        elif audio.shape[1] <= 8:
            audio = np.mean(audio, axis=1)
        else:
            audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def load_audio(audio_path: str, sr: int, audio_seconds: int, clip_mode: str) -> np.ndarray:
    try:
        audio, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        audio, loaded_sr = sf.read(audio_path, dtype="float32")
        audio = normalize_audio_shape(audio)
        if loaded_sr != sr:
            audio = librosa.resample(y=audio, orig_sr=loaded_sr, target_sr=sr)

    audio = truncate_audio(audio, n_seconds=audio_seconds, sample_rate=sr, clip_mode=clip_mode)
    return audio.astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def process_predictions(logits: np.ndarray):
    """
    保留原来的判定逻辑，不改 complete / incomplete / invalid 的判断方式
    """
    out = np.array(logits)

    if not np.isfinite(out).all():
        raise ValueError("Non-finite values detected in model output")

    if out.ndim == 2 and out.shape[1] >= 2:
        row_sums = out.sum(axis=1)
        is_prob = np.all(out >= 0.0) and np.all(out <= 1.0) and np.allclose(row_sums, 1.0, atol=1e-4)
        probs_2d = out if is_prob else softmax_np(out, axis=1)
        preds = np.argmax(probs_2d, axis=1).astype(int)
        return probs_2d, preds

    probs = out.squeeze()
    if np.nanmin(probs) < 0 or np.nanmax(probs) > 1:
        probs = 1.0 / (1.0 + np.exp(-probs))
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs_2d = np.stack([1.0 - probs, probs], axis=1)
    preds = np.argmax(probs_2d, axis=1).astype(int)
    return probs_2d, preds


def build_session(onnx_path: str, use_cuda: bool = False) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)


class AudioClassifierInfer:
    def __init__(
        self,
        onnx_path: str,
        labels: List[str],
        use_cuda: bool = False,
        audio_seconds: int = DEFAULT_AUDIO_SECONDS,
        max_audio_seconds: int = MAX_AUDIO_SECONDS,
        sampling_rate: int = SAMPLING_RATE,
        frontend_conf: Optional[Dict[str, Any]] = None,
        clip_mode: str = DEFAULT_CLIP_MODE,
    ):
        self.onnx_path = onnx_path
        self.labels = labels
        self.use_cuda = use_cuda
        self.audio_seconds = audio_seconds
        self.max_audio_seconds = max_audio_seconds
        self.sampling_rate = sampling_rate
        self.clip_mode = clip_mode

        self.session = build_session(onnx_path, use_cuda=use_cuda)
        self.frontend = AudioFrontend(
            **(frontend_conf or DEFAULT_FRONTEND_CONF)
        )

    def _extract_features(self, audio: np.ndarray):
        feats, feat_len = self.frontend.extract_features(audio)
        feats = np.asarray(feats, dtype=np.float32)[None, ...]
        feat_len = np.asarray([int(feat_len)], dtype=np.int64)
        return feats, feat_len

    def _run_model(self, audio: np.ndarray):
        feats, feat_len = self._extract_features(audio)

        inputs = {x.name for x in self.session.get_inputs()}
        if "feats" not in inputs or "feat_lengths" not in inputs:
            raise RuntimeError(f"ONNX 输入不匹配，实际输入为: {list(inputs)}")

        outputs = self.session.run(None, {
            "feats": feats,
            "feat_lengths": feat_len,
        })
        return outputs[0]

    def predict_file(self, audio_path: str) -> Dict[str, Any]:
        audio = load_audio(
            audio_path,
            sr=self.sampling_rate,
            audio_seconds=self.audio_seconds,
            clip_mode=self.clip_mode,
        )

        logits = self._run_model(audio)
        probs_2d, preds = process_predictions(logits)

        pred_id = int(preds[0])
        prob_list = probs_2d[0].tolist()

        return {
            "audio_path": audio_path,
            "prediction_id": pred_id,
            "prediction": self.labels[pred_id],
            "probabilities": {
                label: float(prob_list[i]) for i, label in enumerate(self.labels)
            },
        }


def main():
    input_wav = DEFAULT_INPUT_WAV
    onnx_path = DEFAULT_ONNX_PATH

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"输入音频不存在: {input_wav}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"模型不存在: {onnx_path}")

    classifier = AudioClassifierInfer(
        onnx_path=onnx_path,
        labels=LABELS,
        use_cuda=False,
        audio_seconds=DEFAULT_AUDIO_SECONDS,
        max_audio_seconds=MAX_AUDIO_SECONDS,
        sampling_rate=SAMPLING_RATE,
        frontend_conf=DEFAULT_FRONTEND_CONF,
        clip_mode=DEFAULT_CLIP_MODE,
    )

    result = classifier.predict_file(input_wav)

    print(result["prediction"])


if __name__ == "__main__":
    main()