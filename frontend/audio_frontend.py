from typing import Tuple
import numpy as np
import kaldi_native_fbank as knf


class AudioFrontend:
    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0.0,
        **kwargs,
    ):
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False

        self.opts = opts
        self.fs = fs
        self.n_mels = n_mels
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.cmvn = self.load_cmvn() if cmvn_file else None

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform.astype(np.float32)
        waveform = waveform * (1 << 15)

        fbank_fn = knf.OnlineFbank(self.opts)
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())

        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins], dtype=np.float32)
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)

        feat = mat.astype(np.float32)
        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        lfr_inputs = []
        t = inputs.shape[0]
        t_lfr = int(np.ceil(t / lfr_n))

        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        t = t + (lfr_m - 1) // 2

        for i in range(t_lfr):
            if lfr_m <= t - i * lfr_n:
                lfr_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                num_padding = lfr_m - (t - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                lfr_inputs.append(frame.reshape(1, -1))

        return np.vstack(lfr_inputs).astype(np.float32)

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars_ = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        return (inputs + means) * vars_

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn is not None:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat.astype(np.float32), feat_len

    def extract_features(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feat, _ = self.fbank(waveform)
        feat, feat_len = self.lfr_cmvn(feat)
        return feat, feat_len

    def load_cmvn(self) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if not line_item:
                continue

            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    means_list = list(line_item[3:(len(line_item) - 1)])
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    vars_list = list(line_item[3:(len(line_item) - 1)])
                    continue

        means = np.array(means_list).astype(np.float64)
        vars_ = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars_])
        return cmvn