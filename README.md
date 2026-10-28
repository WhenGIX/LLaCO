<div align="center">

# LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling

</div>

<div align="center">

Zijian Jia<sup>\*</sup>,
Peixuan Zhang<sup>\*</sup>,
Genying Li,
Di Wu,
Yujie Fu,
Si Li,
Boxin Shi,
Shuchen Weng

<sup>\*</sup>Equal contribution

<a href="https://arxiv.org/abs/2610.32604"><img src="https://img.shields.io/badge/arXiv-2610.32604-b31b1b.svg" alt="arXiv"></a>
<a href="https://escapistmost.github.io/LLaCO/"><img src="https://img.shields.io/badge/Project-Page-0A66C2.svg" alt="Project Page"></a>
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Coming%20Soon-lightgrey" alt="Hugging Face">

</div>

<img src='./assets/pipeline.png' width='100%' />

## Abstract

**LLaCO** is a continuous diffusion framework for large language models, powered by Competition-Aware Semantic Scheduling.

<details><summary>CLICK for the full abstract</summary>
Continuous diffusion models have emerged as a promising non-autoregressive paradigm for text generation, enabling bidirectional contextual planning and fine-grained semantic transitions over continuous embedding spaces. Along the continuous trajectory, the residual token identity in an intermediate state directly shapes the denoising objective at each step. However, the geometric trajectory strictly misaligns with temporal progression, triggering severe temporal compression near the noisy endpoint. In this paper, we show that embedding norm decay causes Euclidean distance to yield spurious overtaking, indicating that token competition must be evaluated directionally. In this space, a candidate overtakes the source token only when the effective noise magnitude surpasses their semantic separation. We formalize these pairwise dynamics as the Competition Threshold Theory (CTT). Crucially, high-dimensional overtaking requires extreme noise levels, causing standard trajectories to concentrate identity decay into a narrow window near the noisy endpoint. To resolve this, we propose Semantic Retention Scheduling (SRS) to align time allocation with the progression of identity decay, and demonstrate that existing token identity metrics are fundamentally statistical aggregations of these pairwise crossings. Building on these theoretical insights, we introduce LLaCO, a generative framework powered by competition-aware semantic scheduling. LLaCO adapts a pretrained backbone to continuous representations via an embedding directional adapter and timestep-conditioned modulation. To operate within the directional space, LLaCO employs a spherical noising process formulated as a geodesic interpolation toward a mask-biased endpoint, eliminating norm decay while preserving accurate token competition. Guided by SRS, LLaCO reallocates the denoising trajectory proportionally to the logarithmic rank decay, dedicating maximum optimization budget to critical identity transitions. Instantiated as an 8B model, LLaCO comprehensively outperforms previous continuous diffusion baselines across extensive benchmarks and rivals discrete diffusion models of comparable scale, laying a principled foundation for future exploration.
</details>

## 💡 News

- 2026.10.28 [LLaCO-8B checkpoint](https://huggingface.co/escapist413/LLaCO-8B), inference code, and training code released.
- 2026.10.27 Paper released on [arXiv](https://arxiv.org/abs/2610.32604).
- 2026.10.23 Project page and technical report released.

## 📑 Todo List

- [x] Release inference code
- [x] Release LLaCO-8B checkpoint
- [x] Release training code

## 📥 Download

Download [**LLaCO-8B**](https://huggingface.co/escapist413/LLaCO-8B) from Hugging Face:

```bash
hf download escapist413/LLaCO-8B --local-dir checkpoints/LLaCO-8B
```

## 🛠️ Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch with CUDA 12.4
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

bash scripts/setup_env.sh
source scripts/env.sh
```

## 🚀 Inference

```bash
source scripts/env.sh

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  -m llaco.infer \
  --checkpoint checkpoints/LLaCO-8B \
  --tokenizer checkpoints/LLaCO-8B/tokenizer \
  --prompt 'A bookstore sells notebooks for $4 each. Emma buys 3 notebooks and pays with a $20 bill. How much change should she receive? Show your reasoning and put the final answer in \boxed{}.' \
  --output outputs/inference.json
```

## 🏋️ Training

First convert the JSONL conversations in `examples/data/train.jsonl` into the prepared dataset format used by training.

```bash
source scripts/env.sh
python scripts/prepare_spherical_data.py \
  --selection examples/data/selection.json \
  --output data/prepared/example \
  --workers 2 \
  --validation-fraction 0
```

This creates `data/prepared/example/manifest.json` and the token shards used by the loader. For your own data, replace `examples/data/train.jsonl` and update `data_root` and `glob` in `selection.json`.

Then train from the released LLaCO-8B checkpoint:

```bash
source scripts/env.sh
export SPHERICAL_TP_SIZE=4 SPHERICAL_DP_SIZE=2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m llaco.train \
  --checkpoint checkpoints/LLaCO-8B \
  --tokenizer checkpoints/LLaCO-8B/tokenizer \
  --prepared-manifest data/prepared/example/manifest.json \
  --output-dir outputs/train \
  --run-name llaco_train \
  --train-iters 10000 \
  --panel-interval 500 \
  --model-only-interval 2000 \
  checkpoint.save_interval=500
```

## Acknowledgements

LLaCO is built on [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), and LLaCO-8B is adapted from the pretrained [Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion) model. We thank the authors for open-sourcing their work.

## Citation

```bibtex
@article{jia2026llaco,
  title   = {LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling},
  author  = {Jia, Zijian and Zhang, Peixuan and Li, Genying and Wu, Di and Fu, Yujie and Li, Si and Shi, Boxin and Weng, Shuchen},
  journal = {arXiv preprint arXiv:2610.32604},
  year    = {2026}
}
```
