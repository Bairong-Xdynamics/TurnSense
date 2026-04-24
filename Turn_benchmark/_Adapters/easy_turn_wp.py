from __future__ import annotations
import os
import sys
from typing import Any, Dict, Optional

from .base import BaseTurnModel

try:
    import torch
except Exception:
    torch = None


class EasyTurnWP(BaseTurnModel):
    """
    Easy-Turn: audio -> {complete,incomplete,backchannel,dismissal}
    关键：模型的 <WAIT> / wait 标签在我们这边对应 dismissal
    """
    model_name = "Easy-Turn"
    supports_audio = True
    supports_text = False
    supports_context = False
    supported_labels = {"complete", "incomplete", "backchannel", "dismissal"}  # ✅ 四分类

    def __init__(self, model_root: str):
        if torch is None:
            raise RuntimeError("Easy-Turn requires torch.")

        import yaml
        import librosa
        import torchaudio

        # 使用传入的 model_root 作为 Easy-Turn 工程根目录，避免硬编码到 Turn_Benchamrk/Easy_Turn
        easy_turn_root = os.path.abspath(model_root)

        if easy_turn_root not in sys.path:
            sys.path.insert(0, easy_turn_root)

        from wenet.utils.init_model import init_model

        self._yaml = yaml
        self._librosa = librosa
        self._torchaudio = torchaudio
        self._init_model = init_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(model_root, "examples/wenetspeech/whisper/conf/train.yaml")
        checkpoint_path = os.path.join(model_root, "pretrained_models/Easy-Turn/checkpoint.pt")
        prompt_yaml = os.path.join(model_root, "examples/wenetspeech/whisper/conf/prompt.yaml")
        llm_path = os.path.join(model_root, "pretrained_models/Qwen2.5-0.5B-Instruct")

        with open(config_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        # 兼容配置文件里写死的旧绝对路径（Turn_Benchamrk/Easy_Turn/...）
        if os.path.isdir(llm_path):
            def _patch_llm_path(node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "llm_path":
                            node[k] = llm_path
                        else:
                            _patch_llm_path(v)
                elif isinstance(node, list):
                    for x in node:
                        _patch_llm_path(x)

            _patch_llm_path(configs)

        class Args:
            checkpoint = checkpoint_path
            config = config_path
            gpu = 0 if torch.cuda.is_available() else -1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = "fp32"
            jit = False

        self.model, _ = init_model(Args(), configs)
        self.model.to(self.device).eval()

        with open(prompt_yaml, "r") as f:
            prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

        # 这里按你现有的 prompt key；如果你有包含 WAIT/DISMISSAL 的 prompt key，可在此替换
        self.prompt = prompt_dict.get("<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>", [""])[0]

    def _load_audio_feature(self, wav_path: str, resample_rate: int = 16000):
        waveform, sample_rate = self._torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != resample_rate:
            waveform = self._torchaudio.transforms.Resample(sample_rate, resample_rate)(waveform)

        n_fft = 400
        hop_length = 160
        num_mel_bins = 80
        window = torch.hann_window(n_fft).to(waveform.device)

        stft = torch.stft(
            waveform.squeeze(0), n_fft, hop_length, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = torch.from_numpy(
            self._librosa.filters.mel(sr=resample_rate, n_fft=n_fft, n_mels=num_mel_bins)
        ).to(magnitudes.device)

        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.transpose(0, 1).to(self.device)

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        wav_path = sample.get("last_wav")
        if not wav_path:
            return None

        feats = self._load_audio_feature(wav_path)
        feats_len = torch.tensor([feats.size(0)], dtype=torch.int32).to(self.device)
        feats = feats.unsqueeze(0)

        with torch.no_grad():
            res = self.model.generate(wavs=feats, wavs_len=feats_len, prompt=self.prompt)

        out = (res[0] if res else "").lower()

        # ✅ 四分类解析：wait -> dismissal
        if "<backchannel>" in out:
            return "backchannel"
        if "<dismissal>" in out or  "<wait>" in out:
            return "dismissal"
        if "<complete>" in out:
            return "complete"
        if "<incomplete>" in out:
            return "incomplete"
        return "incomplete"
