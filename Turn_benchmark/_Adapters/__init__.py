from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    import torchaudio
except Exception:
    torchaudio = None


# ---------------------------
# labels / alias
# ---------------------------
LABELS_4 = ["complete", "incomplete", "backchannel", "dismissal"]
LABEL_SET = set(LABELS_4)

TAG_ALIASES = {
    # completion
    "completion": "complete",
    "complete": "complete",
    "completed": "complete",

    # incomplete
    "incomplete": "incomplete",
    "incompletion": "incomplete",

    # backchannel
    "backchannel": "backchannel",
    "bc": "backchannel",

    # dismissal (wait -> dismissal)
    "wait": "dismissal",
    "dismissal": "dismissal",
    "dismissal_speaking": "dismissal",
    "dismiss": "dismissal",
    "dissmiss": "dismissal",
    "dissmissal": "dismissal",
}

# ---------------------------
# Action mapping (KE-style)
# ---------------------------
ACTIONS_4 = ["takeover", "dismiss", "maintain", "stopandlisten"]

_TAG_RE = re.compile(r"<\|([^|>]+)\|>")


def extract_all_tags(text: str) -> List[str]:
    if not text:
        return []
    return _TAG_RE.findall(text)


def normalize_action_tag(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t in ["stopandlisten", "stop_and_listen", "stoplisten", "stop&listen"]:
        return "stopandlisten"
    if t in ["takeover", "take_over"]:
        return "takeover"
    if t in ["maintain", "keep", "continue"]:
        return "maintain"
    if t in ["dismiss", "wait", "hold"]:
        return "dismiss"
    return t


def action_to_label4(action: str) -> str:
    """
    Action -> label4:
      Maintain      -> backchannel
      StopandListen -> dismissal
      Takeover      -> complete
      Dismiss       -> incomplete
    """
    a = normalize_action_tag(action)
    if a == "maintain":
        return "backchannel"
    if a == "stopandlisten":
        return "dismissal"
    if a == "takeover":
        return "complete"
    if a == "dismiss":
        return "incomplete"
    return ""


def extract_gt_action_and_label4_from_last_assistant(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    GT: 最后一条 assistant.content 形如 "<|Action|><|Semantic|>"
    只用“倒数第二个标签”（Action）做 GT action。
    - 若 tag 数>=2：action = tags[-2]
    - 若 tag 数==1：action = tags[-1]（兜底）
    """
    if not messages:
        return "", ""
    last = messages[-1]
    if (last.get("role") or "").strip() != "assistant":
        return "", ""
    content = (last.get("content") or "").strip()
    if not content:
        return "", ""

    tags = [t.strip() for t in extract_all_tags(content) if str(t).strip()]
    if not tags:
        return "", ""

    action_raw = tags[-2] if len(tags) >= 2 else tags[-1]
    action = normalize_action_tag(action_raw)
    label4 = action_to_label4(action)
    return action, label4


def guess_language(text: str) -> str:
    if not text:
        return "zh"
    ascii_cnt = sum(1 for c in text if ord(c) < 128)
    return "en" if ascii_cnt / max(1, len(text)) > 0.8 else "zh"


def normalize_user_text(text: str) -> str:
    """
    把形如：
      "系统当前状态: <|SystemIdle|> 输入内容：热的"
    解析成：
      "热的"

    兼容一些常见变体，比如：
      "输入内容: xxx"
      "输入: xxx"
      "Input: xxx"
    若解析失败则返回原文本（strip 后）。
    """
    if not text:
        return ""

    t = text.strip()

    markers = [
        "输入内容：", "输入内容:", "输入：", "输入:", "Input:", "Input："
    ]

    for m in markers:
        idx = t.rfind(m)
        if idx != -1:
            cand = t[idx + len(m):].strip()
            if cand:
                return cand

    if "<|System" in t or "系统当前状态" in t:
        for sep in ["：", ":"]:
            if sep in t:
                cand = t.split(sep)[-1].strip()
                if cand:
                    return cand

    return t


def extract_gt_label_from_last_assistant(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    旧 GT 解析（语义 tag）：从最后一条 assistant.content 解析
    <|Completion|> / <|Backchannel|> / <|Dismissal|> / <|Wait|> / <|Incomplete|> ...
    """
    if not messages:
        return None
    last = messages[-1]
    if (last.get("role") or "").strip() != "assistant":
        return None
    content = (last.get("content") or "").strip()
    if not content:
        return None

    tags = re.findall(r"<\|([^|>]+)\|>", content)
    tags = [t.strip().lower() for t in tags if t.strip()]

    for t in reversed(tags):
        if t in TAG_ALIASES:
            return TAG_ALIASES[t]
    return None


def get_last_user_utt(messages: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    last_user = None
    for m in reversed(messages):
        if (m.get("role") or "").strip() == "user":
            last_user = m
            break
    if not last_user:
        return "", None
    audio_path = last_user.get("audio") or last_user.get("wav_path")
    return (last_user.get("content") or "").strip(), audio_path


def build_context(messages: List[Dict[str, Any]], max_history: int = 11) -> List[Dict[str, str]]:
    ctx: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant"):
            continue
        if not content:
            continue
        ctx.append({"role": role, "content": content})
    if max_history > 0:
        ctx = ctx[-max_history:]
    return ctx


# ---------------------------
# helpers
# ---------------------------
def normalize_label(x: Any) -> str:
    """大小写不敏感 + wait->dismissal"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s == "wait":
        return "dismissal"
    if s in ("dismissal_speaking", "dismissal"):
        return "dismissal"
    return s


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_dataset(path: str, default_lang: Optional[str] = None, max_history: int = 6) -> List[Dict[str, Any]]:
    """
    支持 .jsonl / .json(list)
    每条样本要求：messages，且最后一条 assistant 带 tag 作为 GT

    新逻辑：
    1) 优先按 "<|Action|><|Semantic|>" 取倒数第二个 tag 作为 action，并 action->label4
    2) 若取不到 action 或映射失败，再 fallback 到旧语义 tag 解析
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    raw_list: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    raw_list.append(obj)
    elif p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("json dataset must be a list")
        raw_list = [x for x in obj if isinstance(x, dict)]
    else:
        raise ValueError("dataset must be .jsonl or .json")

    src_stem = p.stem
    parsed: List[Dict[str, Any]] = []
    skipped = 0

    for obj in raw_list:
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs:
            skipped += 1
            continue

        gt_action, gt_from_action = extract_gt_action_and_label4_from_last_assistant(msgs)

        if gt_from_action and gt_from_action in LABEL_SET:
            gt = gt_from_action
        else:
            gt = extract_gt_label_from_last_assistant(msgs)

        if gt is None:
            skipped += 1
            continue

        last_text, last_wav = get_last_user_utt(msgs)
        last_text = normalize_user_text(last_text)

        if default_lang in ("zh", "en"):
            lang = default_lang
        else:
            lang = obj.get("language") or guess_language(last_text)

        parsed.append(
            dict(
                raw=obj,
                messages=msgs,
                gt=gt,
                gt_action=gt_action,
                last_text=last_text,
                last_wav=last_wav,
                lang=lang,
                context=build_context(msgs, max_history=max_history),
                _src=src_stem,
                _src_path=str(p),
            )
        )

    if skipped:
        print(f"[WARN] skipped {skipped} samples (missing/invalid messages or GT tag).")
    return parsed


# ---------------------------
# Metrics / perf helpers
# ---------------------------
def get_audio_duration_sec(audio_path: str) -> float:
    if not audio_path or torchaudio is None:
        return 0.01
    try:
        info = torchaudio.info(audio_path)
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return 0.01


def reset_gpu_peak_mem():
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mem_mb() -> float:
    if torch is None:
        return 0.0
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    return 0.0


@dataclass
class Confusion4:
    total: int = 0
    correct: int = 0
    per_class_total: Dict[str, int] = None
    per_class_correct: Dict[str, int] = None
    per_lang_total: Dict[str, int] = None
    per_lang_correct: Dict[str, int] = None

    per_src_class_total: Dict[str, Dict[str, int]] = None
    per_src_class_correct: Dict[str, Dict[str, int]] = None

    per_action_total: Dict[str, int] = None
    per_action_correct: Dict[str, int] = None

    # binary (positive = "complete")
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def __post_init__(self):
        self.per_class_total = {k: 0 for k in LABELS_4}
        self.per_class_correct = {k: 0 for k in LABELS_4}
        self.per_lang_total = {}
        self.per_lang_correct = {}

        self.per_src_class_total = {}
        self.per_src_class_correct = {}

        self.per_action_total = {a: 0 for a in ACTIONS_4}
        self.per_action_correct = {a: 0 for a in ACTIONS_4}

    def update(self, pred: str, gt: str, lang: str, src: Optional[str] = None, gt_action: Optional[str] = None):
        self.total += 1
        ok = (pred == gt)
        if ok:
            self.correct += 1

        if gt in self.per_class_total:
            self.per_class_total[gt] += 1
            if ok:
                self.per_class_correct[gt] += 1

        self.per_lang_total[lang] = self.per_lang_total.get(lang, 0) + 1
        if ok:
            self.per_lang_correct[lang] = self.per_lang_correct.get(lang, 0) + 1

        if src:
            if src not in self.per_src_class_total:
                self.per_src_class_total[src] = {k: 0 for k in LABELS_4}
                self.per_src_class_correct[src] = {k: 0 for k in LABELS_4}
            if gt in self.per_src_class_total[src]:
                self.per_src_class_total[src][gt] += 1
                if ok:
                    self.per_src_class_correct[src][gt] += 1

        if gt_action:
            a = normalize_action_tag(gt_action)
            if a in self.per_action_total:
                self.per_action_total[a] += 1
                if ok:
                    self.per_action_correct[a] += 1

        pred_pos = (pred == "complete")
        gt_pos = (gt == "complete")
        if pred_pos and gt_pos:
            self.tp += 1
        elif pred_pos and (not gt_pos):
            self.fp += 1
        elif (not pred_pos) and gt_pos:
            self.fn += 1
        else:
            self.tn += 1

    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def acc_class(self, c: str) -> Optional[float]:
        n = self.per_class_total.get(c, 0)
        if n == 0:
            return None
        return self.per_class_correct.get(c, 0) / n

    def acc_lang(self, lang: str) -> Optional[float]:
        n = self.per_lang_total.get(lang, 0)
        if n == 0:
            return None
        return self.per_lang_correct.get(lang, 0) / n

    def acc_action(self, a: str) -> Optional[float]:
        a = normalize_action_tag(a)
        n = self.per_action_total.get(a, 0)
        if n == 0:
            return None
        return self.per_action_correct.get(a, 0) / n

    def acc_src_class(self, src: str, c: str) -> Optional[float]:
        if src not in self.per_src_class_total:
            return None
        n = self.per_src_class_total[src].get(c, 0)
        if n == 0:
            return None
        return self.per_src_class_correct[src].get(c, 0) / n

    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return None if d == 0 else self.tp / d

    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return None if d == 0 else self.tp / d

    def f1(self) -> Optional[float]:
        p = self.precision()
        r = self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def fpr(self) -> Optional[float]:
        d = self.fp + self.tn
        return None if d == 0 else self.fp / d

    def fnr(self) -> Optional[float]:
        d = self.fn + self.tp
        return None if d == 0 else self.fn / d


# ---------------------------
# Base model wrapper interface
# ---------------------------
class BaseTurnModel:
    model_name: str = "Base"
    supports_audio: bool = False
    supports_text: bool = True
    supports_context: bool = False
    supported_labels: set = {"complete", "incomplete"}

    def warmup(self, sample: Dict[str, Any], iters: int = 20):
        for _ in range(max(1, iters)):
            _ = self.predict(sample)

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


# ---------------------------
# adapters exports (keep only requested models)
# ---------------------------
from .easy_turn_wp import EasyTurnWP  # noqa: E402
from .smart_turn_wp import SmartTurnWP  # noqa: E402
from .ten_turn_wp import TENTurnWP  # noqa: E402
from .firered_wp import FireRedChatWP  # noqa: E402
from .namo_wp import NAMOTurnWP  # noqa: E402

__all__ = [
    "ACTIONS_4",
    "LABELS_4",
    "LABEL_SET",
    "TAG_ALIASES",
    "BaseTurnModel",
    "Confusion4",
    "EasyTurnWP",
    "FireRedChatWP",
    "NAMOTurnWP",
    "SmartTurnWP",
    "TENTurnWP",
    "action_to_label4",
    "build_context",
    "extract_all_tags",
    "extract_gt_action_and_label4_from_last_assistant",
    "extract_gt_label_from_last_assistant",
    "get_audio_duration_sec",
    "get_last_user_utt",
    "gpu_peak_mem_mb",
    "guess_language",
    "load_dataset",
    "normalize_action_tag",
    "normalize_label",
    "normalize_user_text",
    "now_str",
    "reset_gpu_peak_mem",
]
