from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


THIS_DIR = Path(__file__).resolve().parent  
CODE_ROOT = THIS_DIR.parents[2].resolve()  # code/

TEST_JSONL_DIR = THIS_DIR / "test_jsonl"

# 确保能从本目录加载本地 _Adapters
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _Adapters import (  # noqa: E402
    LABELS_4,
    BaseTurnModel,
    Confusion4,
    EasyTurnWP,
    FireRedChatWP,
    NAMOTurnWP,
    SmartTurnWP,
    TENTurnWP,
    get_audio_duration_sec,
    gpu_peak_mem_mb,
    normalize_label,
    now_str,
    reset_gpu_peak_mem,
)


MODEL_ORDER_BY_LANG: Dict[str, List[str]] = {
    # 中文跑 5 个模型；英文不跑 easy_turn
    "zh": ["easy_turn", "smart_turn", "ten_turn", "firered", "namo"],
    "en": ["smart_turn", "ten_turn", "firered", "namo"],
}


_AUDIO_EXT_CANDIDATES = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"]
DEFAULT_DATASETS_ZH = [
    str(TEST_JSONL_DIR / "easy_test_ZH.jsonl"),
    str(TEST_JSONL_DIR / "ten_test_ZH.jsonl"),
    str(TEST_JSONL_DIR / "semantic_test_ZH.jsonl"),
]

# 二分类场景只测中文；英文数据集不使用


def existing_files(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]


def resolve_audio_path(audio_path: str) -> str:
    """
    尽量容错音频后缀：
    - 原路径存在则直接使用
    - 否则尝试同 stem 的常见后缀（wav/mp3/flac/...）
    """
    p = Path((audio_path or "").strip())
    if not str(p):
        return ""
    if p.exists():
        return str(p)

    stem = p.with_suffix("")
    for ext in _AUDIO_EXT_CANDIDATES:
        cand = stem.with_suffix(ext)
        if cand.exists():
            return str(cand)
        cand_up = stem.with_suffix(ext.upper())
        if cand_up.exists():
            return str(cand_up)
    return str(p)


def save_markdown_report(results: List[Dict[str, Any]], out_path: str):
    if not results:
        print("No results.")
        return
    headers = list(results[0].keys())
    md = []
    md.append("# Turn Detection Benchmark Report\n")
    md.append(f"**Test Date:** {now_str()}\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    md.append("\n> 数据格式: audio_path/text/label。英文不会运行 Easy-Turn。")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(md), encoding="utf-8")
    print(f"\n📄 Report saved to: {out_path}")


def save_json(results: List[Dict[str, Any]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧾 JSON saved to: {out_path}")


def make_run_dir(out_dir: str, lang: str, dataset_tag: str, run_name: str, model_names: List[str]) -> Path:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    rn = run_name.strip() if run_name else "+".join([m.replace(" ", "").replace("/", "_") for m in model_names])[:80]
    run_dir = out_root / f"{ts}_{lang}_{dataset_tag}_{rn}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_simple_jsonl(path: str, default_lang: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() != ".jsonl":
        raise ValueError(f"Only .jsonl is supported for this script: {path}")

    parsed: List[Dict[str, Any]] = []
    skipped = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples > 0 and len(parsed) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue

            gt = normalize_label(obj.get("label"))
            if gt not in LABELS_4:
                skipped += 1
                continue

            parsed.append(
                {
                    "raw": obj,
                    "messages": [],
                    "gt": gt,
                    "gt_action": "",
                    "last_text": (obj.get("text") or "").strip(),
                    "last_wav": resolve_audio_path(obj.get("audio_path") or ""),
                    "lang": default_lang,
                    "context": [],
                    "_src": p.stem,
                    "_src_path": str(p),
                }
            )

    if skipped:
        print(f"[WARN] {p.name}: skipped {skipped} invalid rows.")
    return parsed


def run_benchmark(
    model: BaseTurnModel,
    dataset: List[Dict[str, Any]],
    warmup_iters: int = 20,
    latency_first_n: int = 100,
    save_per_sample_path: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n🚀 Model: {model.model_name}")
    print(
        f"  supports_audio={model.supports_audio}, supports_text={model.supports_text}, "
        f"supports_context={model.supports_context}"
    )
    print(f"  supported_labels={sorted(model.supported_labels)}")

    if dataset:
        print(f"-> warmup {warmup_iters} iters ...")
        try:
            model.warmup(dataset[0], warmup_iters)
        except Exception as e:
            print(f"[WARN] warmup failed: {e}")

    reset_gpu_peak_mem()

    conf = Confusion4()
    latencies: List[float] = []
    infer_latencies: List[float] = []
    infer_total_time: float = 0.0
    infer_samples: int = 0
    rtfs: List[float] = []
    y_true_valid: List[str] = []
    y_pred_valid: List[str] = []
    skipped = 0
    skipped_unsupported_gt = 0

    per_sample_f = None
    if save_per_sample_path:
        Path(save_per_sample_path).parent.mkdir(parents=True, exist_ok=True)
        per_sample_f = open(save_per_sample_path, "w", encoding="utf-8")

    def write_sample(obj: Dict[str, Any]):
        if per_sample_f:
            per_sample_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for idx, s in enumerate(dataset):
        gt_raw = s.get("gt")
        lang = (s.get("lang") or "").strip() or "zh"
        last_text = s.get("last_text") or ""
        last_wav = s.get("last_wav")
        src = s.get("_src")

        gt = normalize_label(gt_raw)

        if gt == "" or gt not in LABELS_4:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_invalid_gt",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "lang": lang,
                    "src": src,
                }
            )
            continue

        if gt not in model.supported_labels:
            skipped += 1
            skipped_unsupported_gt += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_unsupported_gt",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "lang": lang,
                    "src": src,
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            pred_raw = model.predict(s)
            err = None
        except Exception as e:
            pred_raw = None
            err = repr(e)
        t1 = time.perf_counter()

        dt = t1 - t0
        infer_latencies.append(dt)
        infer_total_time += dt
        infer_samples += 1
        audio_len = get_audio_duration_sec(last_wav or "")
        rtf = dt / max(0.01, audio_len)

        pred = normalize_label(pred_raw)
        if pred == "" or pred not in LABELS_4:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_invalid_pred" if err is None else "error",
                    "error": err,
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "pred": pred_raw,
                    "pred_norm": pred,
                    "lang": lang,
                    "src": src,
                    "latency_s": dt,
                    "rtf": rtf,
                }
            )
            continue

        if pred not in model.supported_labels:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_unsupported_pred",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "pred": pred_raw,
                    "pred_norm": pred,
                    "lang": lang,
                    "src": src,
                    "latency_s": dt,
                    "rtf": rtf,
                }
            )
            continue

        if len(latencies) < latency_first_n:
            latencies.append(dt)
        rtfs.append(rtf)
        y_true_valid.append(gt)
        y_pred_valid.append(pred)
        conf.update(pred, gt, lang, src=src, gt_action=None)

        write_sample(
            {
                "idx": idx,
                "model": model.model_name,
                "status": "ok",
                "gt": gt_raw,
                "gt_norm": gt,
                "pred": pred_raw,
                "pred_norm": pred,
                "correct": bool(pred == gt),
                "lang": lang,
                "src": src,
                "latency_s": dt,
                "audio_len_s": audio_len,
                "rtf": rtf,
                "last_text": last_text,
                "last_wav": last_wav,
            }
        )

    if per_sample_f:
        per_sample_f.close()

    avg_lat_ms_first_n = (float(np.mean(latencies)) * 1000.0) if latencies else 0.0
    p50_lat_ms = (float(np.percentile(infer_latencies, 50)) * 1000.0) if infer_latencies else 0.0
    p90_lat_ms = (float(np.percentile(infer_latencies, 90)) * 1000.0) if infer_latencies else 0.0
    avg_lat_ms_all = (float(np.mean(infer_latencies)) * 1000.0) if infer_latencies else 0.0
    throughput = (infer_samples / infer_total_time) if infer_total_time > 0 else 0.0
    avg_rtf = float(np.mean(rtfs)) if rtfs else 0.0
    peak_mem = gpu_peak_mem_mb()

    def fmt_pct(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x*100:.2f}%"

    def one_vs_rest_metrics(cls: str) -> Dict[str, Optional[float]]:
        tp = fp = fn = 0
        for gt_i, pred_i in zip(y_true_valid, y_pred_valid):
            if pred_i == cls and gt_i == cls:
                tp += 1
            elif pred_i == cls and gt_i != cls:
                fp += 1
            elif pred_i != cls and gt_i == cls:
                fn += 1
        precision = None if (tp + fp) == 0 else tp / (tp + fp)
        recall = None if (tp + fn) == 0 else tp / (tp + fn)
        f1 = None if (precision is None or recall is None or (precision + recall) == 0) else 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    m_complete = one_vs_rest_metrics("complete")
    m_incomplete = one_vs_rest_metrics("incomplete")

    def mean_valid(values: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in values if x is not None]
        if not xs:
            return None
        return float(np.mean(xs))

    # 仅保留你指定的指标字段（其余全部不写入 results/report）
    result: Dict[str, Any] = {
        "model": model.model_name,
        "total": conf.total,
        "precision_complete": fmt_pct(m_complete["precision"]),
        "recall_complete": fmt_pct(m_complete["recall"]),
        "f1_complete": fmt_pct(m_complete["f1"]),
        "precision_incomplete": fmt_pct(m_incomplete["precision"]),
        "recall_incomplete": fmt_pct(m_incomplete["recall"]),
        "f1_incomplete": fmt_pct(m_incomplete["f1"]),
        "p50_latency_ms": f"{p50_lat_ms:.2f}",
        "p90_latency_ms": f"{p90_lat_ms:.2f}",
    }

    print("\n✅ Result (selected metrics):")
    for k in result.keys():
        print(f"  {k}: {result[k]}")

    return result


def _pick_lang_path(paths: Dict[str, Dict[str, str]], key: str, lang: str) -> str:
    if key not in paths:
        raise KeyError(f"Missing model path config: {key}")
    if lang not in paths[key]:
        raise KeyError(f"Missing language path config: {key}.{lang}")
    return paths[key][lang]


def _resolve_ten_device() -> str:
    # 按需求：TEN-Turn 固定使用 GPU
    return "cuda"


def _ensure_easy_turn_import_path(paths: Dict[str, Dict[str, str]]):
    """
    Easy-Turn 依赖 `wenet` 包，确保其工程根目录在 sys.path 里。
    """
    easy_root = (paths.get("easy_turn", {}) or {}).get("root", "")
    if not easy_root:
        return
    p = Path(easy_root).resolve()
    if not p.exists():
        return
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def build_models_for_lang(paths: Dict[str, Dict[str, str]], lang: str) -> List[BaseTurnModel]:
    lang = (lang or "").strip().lower()
    if lang not in MODEL_ORDER_BY_LANG:
        raise ValueError(f"Unsupported lang: {lang}")

    _ensure_easy_turn_import_path(paths)
    ten_device = _resolve_ten_device()

    builders: Dict[str, Callable[[], BaseTurnModel]] = {
        "easy_turn": lambda: EasyTurnWP(paths["easy_turn"]["root"]),
        # 按需求：其余模型走 CPU
        "smart_turn": lambda: SmartTurnWP(paths["smart_turn"]["onnx"], prefer_gpu=False),
        "ten_turn": lambda: TENTurnWP(paths["ten_turn"]["model"], device=ten_device),
        "firered": lambda: FireRedChatWP(_pick_lang_path(paths, "firered", lang), lang=lang, threshold=0.5, use_gpu=False),
        "namo": lambda: NAMOTurnWP(_pick_lang_path(paths, "namo", lang), filename="model_quant.onnx", prefer_gpu=False),
    }

    models: List[BaseTurnModel] = []
    for model_key in MODEL_ORDER_BY_LANG[lang]:
        try:
            models.append(builders[model_key]())
        except Exception as e:
            print(f"[WARN] Skip model {model_key}: init failed -> {e}")
            continue
    if not models:
        raise RuntimeError(f"No model initialized successfully for lang={lang}")
    return models


def run_one_lang(
    out_dir: str,
    run_name: str,
    lang: str,
    dataset_paths: List[str],
    paths: Dict[str, Dict[str, str]],
    max_samples_per_file: int = 0,
):
    print("\n==============================")
    print(f"🧪 Running LANG={lang}  files={len(dataset_paths)}")
    print("==============================")

    models_to_run = build_models_for_lang(paths, lang)
    model_names = [m.model_name for m in models_to_run]
    dataset_tag = "+".join([Path(p).stem for p in dataset_paths])[:80]
    run_dir = make_run_dir(out_dir, lang, dataset_tag, run_name, model_names)
    print(f"\n📦 Run output dir: {run_dir}")

    results: List[Dict[str, Any]] = []
    for p in dataset_paths:
        dataset = load_simple_jsonl(p, default_lang=lang, max_samples=max_samples_per_file)
        dataset_name = Path(p).stem
        print(f"\n📚 Dataset: {dataset_name}  samples={len(dataset)}")

        for m in models_to_run:
            per_sample_path = str(run_dir / f"per_sample__{dataset_name}__{m.model_name.replace('/', '_')}__{lang}.jsonl")
            result = run_benchmark(m, dataset, save_per_sample_path=per_sample_path)
            result["dataset"] = dataset_name
            results.append(result)

    save_markdown_report(results, str(run_dir / "report.md"))
    save_json(results, str(run_dir / "results.json"))

    config = {
        "LANG": lang,
        "datasets": dataset_paths,
        "out_dir": out_dir,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "models": model_names,
        "timestamp": now_str(),
        "paths": paths,
        "note": "Simple jsonl benchmark: audio_path/text/label",
    }
    Path(run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧩 Config saved to: {run_dir / 'config.json'}")


def build_default_paths(base_dir: Path) -> Dict[str, Dict[str, str]]:
    return {
        "easy_turn": {
            "root": str((CODE_ROOT / "LLM" / "Easy_Turn").resolve()),
        },
        "smart_turn": {
            "onnx": str(
                (CODE_ROOT / "LLM" / "Smart_Turn_v3" / "pretrained_models" / "smart-turn-v3" / "smart-turn-v3.0.onnx").resolve()
            ),
        },
        "ten_turn": {
            "model": str(
                (CODE_ROOT / "LLM" / "TEN_Turn_Detection" / "pretrained_models" / "TEN_Turn_Detection").resolve()
            ),
        },
        "firered": {
            "zh": str(
                (CODE_ROOT / "benchmark_code" / "FireRedChat" / "models" / "FireRedChat-turn-detector").resolve()
            ),
            "en": str(
                (CODE_ROOT / "benchmark_code" / "FireRedChat" / "models" / "FireRedChat-turn-detector").resolve()
            ),
        },
        "namo": {
            "zh": str(
                (CODE_ROOT / "benchmark_code" / "NAMO-Turn-Detector-v1" / "models" / "Namo-Turn-Detector-v1-Chinese").resolve()
            ),
            "en": str(
                (CODE_ROOT / "benchmark_code" / "NAMO-Turn-Detector-v1" / "models" / "Namo-Turn-Detector-v1-English").resolve()
            ),
        },
    }


def _parse_dataset_arg(v: str) -> List[str]:
    if not v.strip():
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=str((THIS_DIR / "benchmark_runs").resolve()),
        help="output root dir",
    )
    parser.add_argument("--run_name", default="", help="optional short name for this run")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (for reproducibility; affects TEN-Turn sampling)",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=0,
        help="debug mode: limit loaded samples per dataset file (0 means all)",
    )
    args = parser.parse_args()

    # 全局随机种子，确保 do_sample 类模型每次可复现
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # 保持尽量确定性（不强制 use_deterministic_algorithms，避免不兼容报错）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    # 显式写死要测试的数据集（只跑中文）
    zh_paths = existing_files(DEFAULT_DATASETS_ZH)

    if not zh_paths:
        raise RuntimeError("No zh dataset files found. Please check the test_jsonl/*.jsonl in this folder.")

    print(f"[DATA] zh files={len(zh_paths)}")
    for p in zh_paths:
        print(f"  - {p}")

    base_dir = CODE_ROOT
    paths = build_default_paths(base_dir)

    run_one_lang(
        args.out_dir,
        args.run_name,
        "zh",
        zh_paths,
        paths,
        max_samples_per_file=max(0, args.max_samples_per_file),
    )


if __name__ == "__main__":
    main()
