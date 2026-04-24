<div align="center">

<img src="./image/Baiji_Team.png" alt="Baiji Team Logo" width="1000" height="500"/>

<br/>

# TurnSense

### 🎯 轻量 · 精准 · 三分类 — 重新定义语音轮次判别

<br/>

<center><strong>47M 参数 ｜ CPU 延迟 ~55ms ｜ F1 高达 96.35% ｜ 无效语义过滤</strong></center>

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-brgroup--Team/TurnSense-181717?style=for-the-badge&logo=github)](https://github.com/Bairong-Xdynamics/TurnSense)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-brgroup--Team-yellow?style=for-the-badge)](https://huggingface.co/brgroup/TurnSense)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](https://github.com/Bairong-Xdynamics/TurnSense)

</div>

<br/>

**语言**: [English](./README_en.md) | **中文**

<br/>

> **⭐ 如果 TurnSense 对你有帮助，请给我们一个 Star！** 这将帮助我们持续改进模型与文档。

<br/>

## 📖 目录

- [为什么选择 TurnSense](#-为什么选择-turnsense)
- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [模型参数量对比](#-模型参数量对比)
- [基准测试结果](#-基准测试结果)
- [快速开始](#-快速开始)
- [评测说明](#-评测说明)
- [引用](#-引用)
- [问题与交流](#-问题与交流)
- [许可证](#-许可证)

<br/>

---

<br/>

## 🏆 为什么选择 TurnSense

<div align="center">

| 维度 | TurnSense 表现 |
| :---: | :---: |
| 🎯 **准确率** | F1 **96.35%**（easyturn_real_test_ZH）— 同类最优 |
| ⚡ **推理延迟** | CPU p50 ≈ **54.65ms** — 满足实时交互需求 |
| 📦 **模型体积** | 仅 **47M** 参数，INT8 版本仅 **~50MB** |
| 🧠 **分类能力** | 业内首个同时支持 **complete / incomplete / invalid** 三分类 |
| 🚫 **无效过滤** | 无效语义 F1 达 **94.34%**，有效抑制噪声误触发 |
| 🤗 **开源友好** | 提供 FP32 / INT8 ONNX，开箱即用 |

</div>

<br/>

---

<br/>

## 📌 项目简介

**TurnSense** 是一个面向人机语音交互场景的 **三分类语义判别模型**，专注解决对话系统中一个核心问题：

> **用户说话过程中，系统应该立即响应，还是继续等待？**

传统方案通常只做"是否结束"的二分类判断。**TurnSense 更进一步** — 它同时建模语义完整度与无效输入识别，帮助系统在复杂真实场景下实现更自然的轮次衔接，**大幅减少误打断、抢话和无效触发**。

<div align="center">
  <img src="./image/TurnSense.svg" alt="TurnSense 三分类示意图" width="820"/>
</div>

<br/>

TurnSense 将用户输入划分为三种语义状态：

| 状态 | 含义 | 示例 |
| :---: | :--- | :--- |
| ✅ **完整语义 (complete)** | 用户表达已形成完整意图，系统可以响应 | `"帮我查一下明天上海天气。"` |
| ⏳ **不完整语义 (incomplete)** | 用户表达尚未完成，存在截断或停顿后续 | `"我想问一下那个订单就是昨天……"` |
| 🔇 **无效语义 (invalid)** | 输入不构成有效语义，不应触发响应 | `"…（持续噪声 / 非语义发声）"` |

这三类标签让系统不仅能判断 **"是否该接话"**，还能识别 **"是否值得接话"**，从而在语音助手、实时通话、智能客服等场景中显著提升交互自然度与系统稳定性。

<br/>

---

<br/>

## ✨ 核心特性

### 🧠 语义级三分类

同时建模 `complete / incomplete / invalid` 三种状态，比传统二分类更贴近真实对话行为，也是目前**开源模型中唯一原生支持无效语义检测**的方案。

### ⚡ 极致轻量，极速推理

仅 **47M** 参数（INT8 版本约 50MB），CPU 环境下推理延迟 p50 ≈ **54.65ms**、p90 ≈ **58.00ms** — 无需 GPU 即可满足实时交互的严苛要求。

### 🎯 精度领先

在 easyturn_real_test_ZH（300 条）上取得 **F1 96.35%**（complete）和 **F1 96.32%**（incomplete），在 semantic_test_ZH（2000 条）上取得 **F1 92.30%**（complete）和 **F1 91.62%**（incomplete），均为同类最优或次优水平。

### 🚫 无效输入过滤

在 NonverbalVocalization 测试集上，无效语义识别的 precision 达 **100%**、recall 达 **90.37%**（F1 = 94.34%），有效抑制非语义发声和噪声带来的误触发。

### ⚖️ 更稳健的轮次决策

在语义模糊、停顿或口语化表达场景下，兼顾 precision 与 recall，减少过早响应和漏响应。

### 📊 可复现评测体系

配套完整评测流程与脚本，支持统一指标对比、性能回归分析，确保实验可复现。

### 🤗 开源友好，即插即用

标准化仓库结构，提供 FP32 / INT8 ONNX 模型，从安装到推理只需几分钟。

<br/>

---

<br/>

## 📐 模型参数量对比

<div align="center">

| 模型 | 参数量 | 三分类 | 链接 |
| :--- | :---: | :---: | :--- |
| TEN-Turn | **7B**（70 亿） | ❌ | [TEN-framework/TEN_Turn_Detection](https://huggingface.co/TEN-framework/TEN_Turn_Detection) |
| Easy-Turn | 850M | ❌ | [ASLP-lab/Easy-Turn](https://huggingface.co/ASLP-lab/Easy-Turn) |
| NAMO-Turn-Detector (ZH) | 66M | ❌ | [videosdk-live/Namo-Turn-Detector-v1-Multilingual](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual) |
| **⭐ TurnSense** | **47M** | **✅** | [**Baiji-Team/TurnSense**](https://huggingface.co/brgroup/TurnSense) |
| Smart-Turn-v3 | 8M | ❌ | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) |
| FireRedChat-turn-detector | -- | ❌ | [FireRedTeam/FireRedChat-turn-detector](https://huggingface.co/FireRedTeam/FireRedChat-turn-detector) |

</div>

> 💡 TurnSense 以仅 **47M** 的参数量实现了三分类能力，在精度与体积之间取得了最优平衡。

<br/>

---

<br/>

## 📊 基准测试结果

> 以下所有结果均基于开源中文评测集。延迟标注 `(GPU)` 表示 GPU 环境评测，未标注则为 **CPU 环境**。

<br/>

### 📋 easyturn_real_test_ZH（300 条）

> 数据来源：[Easy-Turn-Testset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) 真实数据样本

| 模型 | P (complete) | R (complete) | **F1 (complete)** | P (incomplete) | R (incomplete) | **F1 (incomplete)** | p50 延迟 | p90 延迟 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Easy-Turn | 97.26% | 94.67% | 95.95% | 94.81% | 97.33% | 96.05% | 183.87 (GPU) | 300.37 (GPU) |
| Smart-Turn-v3 | 64.97% | 76.67% | 70.34% | 71.54% | 58.67% | 64.47% | 36.84 | 39.10 |
| TEN-Turn | **99.25%** | 88.00% | 93.29% | 89.22% | **99.33%** | 94.01% | 17.66 (GPU) | 19.41 (GPU) |
| FireRedChat | 70.65% | 94.67% | 80.91% | 91.92% | 60.67% | 73.09% | 98.30 | 99.42 |
| NAMO-Turn | 81.53% | 85.33% | 83.39% | 84.62% | 80.67% | 82.59% | 3.60 | 83.44 |
| **⭐ TurnSense** | 96.03% | **96.67%** | **🏆 96.35%** | **96.64%** | 96.00% | **🏆 96.32%** | 54.65 | 58.00 |

> **🔍 关键发现：** TurnSense 在 complete 和 incomplete 两类上均取得 **最高 F1**，且是唯一在 CPU 上 p50 < 60ms 同时 F1 > 96% 的模型。

<br/>

### 📋 semantic_test_ZH（2000 条）

> 数据来源：[KE-Team/SemanticVAD-Dataset](https://huggingface.co/datasets/KE-Team/SemanticVAD-Dataset) 中文测试集

| 模型 | P (complete) | R (complete) | **F1 (complete)** | P (incomplete) | R (incomplete) | **F1 (incomplete)** | p50 延迟 | p90 延迟 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Easy-Turn | 78.14% | 98.30% | 87.07% | 97.64% | 70.30% | 81.74% | 183.87 (GPU) | 300.37 (GPU) |
| Smart-Turn-v3 | 59.25% | 88.10% | 70.85% | 76.80% | 39.40% | 52.08% | 36.84 | 39.10 |
| TEN-Turn | 85.25% | **99.60%** | 91.87% | **99.52%** | 82.70% | 90.33% | 17.66 (GPU) | 19.41 (GPU) |
| FireRedChat | 66.76% | 99.40% | 79.87% | 98.83% | 50.50% | 66.84% | 98.30 | 99.42 |
| NAMO-Turn | 71.48% | 86.70% | 78.36% | 83.10% | 65.40% | 73.20% | 3.60 | 83.44 |
| **⭐ TurnSense** | **88.96%** | 95.90% | **🏆 92.30%** | 95.55% | **88.00%** | **🏆 91.62%** | 54.65 | 58.00 |

> **🔍 关键发现：** 在 2000 条的大规模测试集上，TurnSense 依然保持 F1 最优，验证了模型的泛化能力。

<br/>

### 📋 NonverbalVocalization_invalid（728 条）

> 数据来源：OpenSLR [Deeply Nonverbal Vocalization Dataset（SLR99）](https://openslr.elda.org/99/)

| 模型 | P (invalid) | R (invalid) | **F1 (invalid)** |
| :--- | :---: | :---: | :---: |
| **⭐ TurnSense** | **100.00%** | **90.37%** | **🏆 94.34%** |

> **🔍 关键发现：** 目前仅 TurnSense 支持无效语义判别。precision 达到 **100%** 意味着零误报，有效防止噪声触发系统响应。

<br/>

---

<br/>

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/Bairong-Xdynamics/TurnSense.git
cd TurnSense

pip install -U numpy onnxruntime torch librosa soundfile pandas scikit-learn huggingface_hub
```

### 2. 获取模型权重

TurnSense 模型权重已发布在 Hugging Face：[Baiji-Team/TurnSense](https://huggingface.co/brgroup/TurnSense)

| 版本 | 体积 | 适用场景 |
| :--- | :--- | :--- |
| FP32 | ~191 MB | 精度优先 |
| INT8 | ~50 MB | 部署优先（推荐） |

**下载方式：**

**方式一：自动下载（推荐）**
推理脚本内置了 Hugging Face 下载逻辑，首次运行时会自动拉取并缓存模型。

**方式二：Git LFS**

```bash
git lfs install
git clone https://huggingface.co/brgroup/TurnSense
```

**方式三：Hugging Face Hub**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="brgroup/TurnSense")
```

### 3. 推理

```bash
python infer.py
```

示例输出：

```
Loading model from brgroup/TurnSense...
Running inference on: "我想问一下那个订单就是昨天..."

Results:
  Input: "我想问一下那个订单就是昨天..."
  TurnSense Detection Result: "incomplete"
```

<br/>

---

<br/>

## 🧪 评测说明

### 1）评测流程

1. 读取 `.jsonl` 格式的测试数据集
2. 每个模型先进行预热（默认 `warmup_iters=20`）
3. 逐样本推理，统计分类指标与性能指标
4. 自动输出汇总与明细文件

输出文件包括：

| 文件 | 说明 |
| :--- | :--- |
| `report.md` | 汇总评测报告 |
| `results.json` | 结构化评测结果 |
| `config.json` | 本次评测配置 |
| `per_sample__*.jsonl` | 逐条样本预测结果 |

### 2）数据格式要求（JSONL）

每一行是一个 JSON 对象，至少包含以下字段：

| 字段 | 说明 |
| :--- | :--- |
| `audio_path` | 音频文件路径 |
| `text` | 文本内容 |
| `label` | 标签（`complete` / `incomplete` / `invalid`） |

示例：

```jsonl
{"audio_path":"/001.wav","text":"帮我查一下明天上海天气","label":"complete"}
{"audio_path":"/002.wav","text":"我想问一下那个订单就是昨天...","label":"incomplete"}
{"audio_path":"/003.wav","text":"啊…嗯…（持续噪声）","label":"invalid"}
```

### 3）运行评测

```bash
python TurnSense/Turn_benchmark/benchmark.py
```

<br/>

---

<br/>

## 📚 引用

如果你在研究或产品中使用了 TurnSense，请引用：

```bibtex
@misc{turnsense2026,
  author       = {Baiji Team},
  title        = {TurnSense: A Three-Class Semantic Detection Model for Complete, Incomplete, and Invalid Utterances},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/brgroup/TurnSense}},
}
```

<br/>

<br/>

## ❓ 问题与交流

如果有问题或改进建议，欢迎通过以下方式联系我们：

| 渠道 | 联系方式 |
| :--- | :--- |
| 📧 邮箱 | [huan.shen@brgroup.com](mailto:huan.shen@brgroup.com) ・ [yingao.wang@brgroup.com](mailto:yingao.wang@brgroup.com) ・ [wei.zou@brgroup.com](mailto:wei.zou@brgroup.com) |
| 💬 微信 | h2538406363 |
| 👥 微信群聊 | 扫码加入群聊<br><img src="TurnSense/image/wechat.jpg" alt="微信群聊二维码" width="220" /> |
| 🐛 Issues | [GitHub Issues](https://github.com/Bairong-Xdynamics/TurnSense/issues) |
| 🔀 PR | [Pull Requests](https://github.com/Bairong-Xdynamics/TurnSense/pulls) |

<br/>

## 📄 许可证

本项目基于 **Apache License 2.0** 发布，并附加特定限制条件。详情请参见 [LICENSE](./LICENSE)。

<br/>

---

<div align="center">

**由 [Baiji Team](https://github.com/Bairong-Xdynamics) 用 ❤️ 打造**

</div>
