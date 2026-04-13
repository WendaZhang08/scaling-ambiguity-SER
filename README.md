# Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models

[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.14620-b31b1b.svg)](https://arxiv.org/abs/2601.14620)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

Official implementation of the ICASSP 2026 paper:

> **Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models**
> Wenda Zhang, Hongyu Jin\*, Siyi Wang\*, Zhiqiang Wei, Ting Dang
> *University of Melbourne & Xi'an Jiaotong University*
> (\* Equal contribution)

## Overview

Ambiguous Emotion Recognition (AER) represents emotions as probability distributions over categories rather than single categorical labels, better capturing the inherently subjective nature of human emotion. However, the reliability of these distributions is fundamentally limited by sparse human annotations — typically only 3–5 ratings per utterance.

This work investigates whether **Large Audio-Language Models (ALMs)** can mitigate this annotation bottleneck. We propose a three-component framework:

- **Synthetic Perceptual Proxies** — ALMs generate diverse synthetic emotion annotations by varying sampling temperature and annotator persona prompts, which are combined with human labels to enrich emotion distributions.
- **DiME-Aug** — A Distribution-aware Multimodal Emotion Augmentation strategy that addresses class imbalance by interpolating minority-class samples in the multimodal (audio + text) feature space.
- **ALM Fine-tuning** — Qwen2-Audio is fine-tuned with LoRA and a distributional classifier head, optimized directly with Jensen-Shannon Divergence loss.

Experiments on **IEMOCAP** and **MSP-Podcast** show that synthetic annotations effectively complement human labels in low-ambiguity settings, while human annotations remain indispensable for highly ambiguous emotions.

---

## Repository Structure

```
scaling-ambiguity-ser/
│
├── approximated_distribution_generation/
│   ├── synthetic_annotation_generation.ipynb   # Generate synthetic annotations via Gemini 2.5 Pro
│   └── prepare_approximated_distributions.ipynb # Aggregate raw annotations into distributions
│
├── data/
│   └── data_processing.ipynb                   # Dataset preprocessing and split preparation
│
├── evaluation/
│   ├── saturation_analysis.ipynb               # Figure 2: JS divergence vs. annotation count
│   ├── ambiguity_level_analysis.ipynb          # Figure 3: Performance by ambiguity level
│   ├── fig2a.pdf, fig2b.pdf                    # Saturation analysis figures
│   └── fig3a.pdf, fig3b.pdf                    # Ambiguity level figures
│
├── lib/
│   ├── __init__.py
│   ├── load_data.py                            # Dataset loading utilities
│   └── evaluation_lib.py                       # Evaluation metrics (JS divergence, BC, etc.)
│
├── model_finetuning/
│   ├── train.py                                # Fine-tuning script (all 6 configurations)
│   └── evaluate.py                             # Inference and evaluation script
│
├── processed_data/                             # Processed annotations (not included, see Data)
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/WendaZhang08/scaling-ambiguity-ser.git
cd scaling-ambiguity-ser
```

### 2. Install PyTorch with CUDA support

Install PyTorch separately following the [official guide](https://pytorch.org/get-started/locally/). For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the base model

Download [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) and place it under `model_finetuning/models/`:

```bash
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2-Audio-7B-Instruct",
    local_dir="model_finetuning/models/Qwen2-Audio-7B-Instruct"
)
```

---

## Data

This work uses two publicly available datasets that require access requests from their respective providers:

| Dataset | Description | Access |
|---------|-------------|--------|
| [IEMOCAP](https://sail.usc.edu/iemocap/) | Dyadic actor conversations, 3 annotators/utterance | Request from USC SAIL |
| [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) | Naturalistic podcast speech, 5–21 annotators/utterance | Request from UT Dallas MSP Lab |

Once obtained, place the datasets under the `data/` directory and follow `data/data_processing.ipynb` for preprocessing. We focus on four emotion categories: **Angry**, **Happy**, **Neutral**, **Sad** (label conventions differ slightly between datasets — see the notebook for details).

---

## Synthetic Annotation Generation

Synthetic annotations are generated using **Gemini 2.5 Pro** via the Google GenAI API. The generation pipeline is in `approximated_distribution_generation/synthetic_annotation_generation.ipynb`.

Each utterance is annotated multiple times with varied sampling temperatures (0.1–1.0) and annotator persona prompts to simulate diverse human perspectives. The resulting single-label annotations are then aggregated into probability distributions in `prepare_approximated_distributions.ipynb`.

A Gemini API key is required:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

---

## Training

The unified training script supports all six experimental configurations via command-line arguments:

```bash
# IEMOCAP experiments
python model_finetuning/train.py --dataset iemocap --annotation_source human
python model_finetuning/train.py --dataset iemocap --annotation_source synthetic
python model_finetuning/train.py --dataset iemocap --annotation_source combined

# MSP-Podcast experiments
python model_finetuning/train.py --dataset msp --annotation_source human
python model_finetuning/train.py --dataset msp --annotation_source synthetic
python model_finetuning/train.py --dataset msp --annotation_source combined
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `iemocap` or `msp` |
| `--annotation_source` | required | `human`, `synthetic`, or `combined` |
| `--batch_size` | `8` | Per-device training batch size |
| `--epochs` | `60` | Maximum training epochs (early stopping patience: 8) |
| `--dime_ratio` | `0.30` | DiME-Aug target ratio relative to majority class |

### Training configuration

Key hyperparameters follow Section 3.2 of the paper:

- **Backbone**: Qwen2-Audio-7B-Instruct
- **LoRA**: rank `r=8`, alpha `α=16`, dropout `0.2`, applied to Q/K/V/O projections
- **Optimizer**: AdamW with cosine learning rate scheduling, `lr=2.5e-6`
- **Loss**: Jensen-Shannon Divergence with uniformity penalty
- **Effective batch size**: 64 (batch size 8 × gradient accumulation steps 8)

---

## Evaluation

```bash
# IEMOCAP experiments
python model_finetuning/evaluate.py --dataset iemocap --annotation_source human
python model_finetuning/evaluate.py --dataset iemocap --annotation_source synthetic
python model_finetuning/evaluate.py --dataset iemocap --annotation_source combined

# MSP-Podcast experiments
python model_finetuning/evaluate.py --dataset msp --annotation_source human
python model_finetuning/evaluate.py --dataset msp --annotation_source synthetic
python model_finetuning/evaluate.py --dataset msp --annotation_source combined
```

To evaluate a specific checkpoint:

```bash
python model_finetuning/evaluate.py --dataset iemocap --annotation_source combined --checkpoint checkpoint-1150
```

Results are saved to `model_finetuning/evaluation_results_{dataset}_{annotation_source}/`.

---

## Results

Performance on IEMOCAP and MSP-Podcast test sets (JS Divergence ↓, Bhattacharyya Coefficient ↑):

**IEMOCAP**

| Annotation Source | w/ DiME-Aug | JS ↓ | BC ↑ |
|-------------------|:-----------:|------|------|
| Human-only        | ✓ | **0.302** | **0.724** |
| Synthetic-only    | ✓ | 0.431 | 0.607 |
| Combined          | ✓ | 0.325 | 0.715 |

**MSP-Podcast**

| Annotation Source | w/ DiME-Aug | JS ↓ | BC ↑ |
|-------------------|:-----------:|------|------|
| Human-only        | ✓ | 0.307 | 0.719 |
| Synthetic-only    | ✓ | 0.373 | 0.660 |
| Combined          | ✓ | **0.274** | **0.757** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhang2026scalingambiguityaugmentinghuman,
      title={Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models}, 
      author={Wenda Zhang and Hongyu Jin and Siyi Wang and Zhiqiang Wei and Ting Dang},
      year={2026},
      eprint={2601.14620},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2601.14620}, 
}
```

The official IEEE Xplore citation will be available after the conference proceedings are published.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
