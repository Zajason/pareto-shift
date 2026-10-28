# pareto-shift

**Does hardware change which generative model is efficient?**
Training-free acceleration of diffusion and flow-matching transformers across GPU generations.

Project for *Advanced Topics in Artificial Intelligence* (NTUA ECE, 2026–27). Author: Iason-Daniil Zakynthinos.

---

## Overview

Sampling efficiency of image generators is usually reported as the number of function evaluations (NFE) or as a
timing on one accelerator. NFE is a sound proxy only while every network call costs the same. The training-free
accelerations used in practice, **feature caching** and **low-precision (INT8/FP8) kernels**, change the cost of
a call without changing NFE, and how much they help depends on what the hardware supports.

This project measures whether quality–cost rankings of **diffusion (DiT)** and **flow-matching (SiT)** samplers,
with and without these accelerations, transfer across three GPU generations. DiT and SiT share the same
architecture and GFLOPs, so they form a controlled pair. No model is trained.

**Research question.** When an acceleration changes the cost of a network call but not the number of calls, do
the resulting quality–cost rankings of diffusion and flow-matching samplers survive a change of GPU?

| | Hypothesis |
|---|---|
| **H1 — Control** | Without acceleration, latency is linear in NFE on every device and rankings are stable. This validates the harness and isolates the effect of the accelerations. |
| **H2 — Rankings move** | Speedups from caching and INT8/FP8 depend on the device and batch size, so adding them lowers cross-device rank agreement and changes Pareto-frontier membership. Neither NFE nor FLOPs predicts this. |
| **H3 — Who tolerates caching** | Caching hurts least where consecutive network outputs are most similar. Flow matching, with straighter paths, should lose less quality under caching than diffusion, and both should lose more at 8 steps than at 32. At 8 steps caching is predicted to fail, in quality, speed or both: a stated prediction to test, not a risk. |

Either outcome is informative: it shows when NFE or FLOPs can stand in for measured cost, and when an efficiency
claim must state its hardware.

## Experimental design

A full factorial of **2 × 2 × 3 × 4 = 48** configurations:

| Factor | Diffusion · DiT-XL/2 | Flow · SiT-XL/2 | Levels |
|---|---|---|---|
| Checkpoint | ImageNet-256, public | ImageNet-256, public | 2 |
| Sampler | DDIM · DPM-Solver | Euler · Heun (ODE) | 2 each |
| Steps | 8 · 16 · 32 | 8 · 16 · 32 | 3 |
| Acceleration | none (fp16) · feature caching (FORA-style) · INT8 W8A8 · FP8 W8A8 | same | 4 |
| Batch size | 1 · 16 (timing only) | 1 · 16 (timing only) | — |

Fixed throughout: classifier-free guidance scale, compile settings, seeds and class labels. Every network call
counts toward NFE. FLOPs per image are counted separately, so cached calls show up as cheaper calls.

### Hardware

| Device | Architecture | Role |
|---|---|---|
| NVIDIA RTX 5080 (local) | Blackwell; fp16, bf16, INT8, FP8 | Main machine: all image generation, pilot sweeps |
| Google Colab T4 | Turing; fp16, INT8 (no bf16, no FP8) | Timing sweeps |
| Ikaros GH200 | Hopper; fp16, bf16, INT8, FP8 (limited hours) | Timing sweeps and validation |

Each run uses one GPU. A kernel smoke test in week 1 runs fp16, INT8 and FP8 on each GPU. If a device has no
kernel for a configuration (e.g. FP8 on the T4), the configuration is fixed as unavailable before the sweep.
That gap is part of the result.

### Measurement protocol (identical on every device)

1. **Pin** software versions, seeds and class labels.
2. **Warm up** kernels before any timing.
3. **Time** with repeated CUDA-event runs.
4. **Measure** peak memory and energy from the NVML counter. On the GH200, check that it reports GPU power rather than whole-module power.
5. **Report** median and interquartile range.

### Software stack

The stack can't be fully identical across devices: Colab fixes its own environment, the RTX 5080 needs a recent
CUDA release, and the GH200 host is ARM. To keep hardware the only intended variable:

- every method uses one in-house implementation;
- the same PyTorch and torchao versions are pinned wherever they are available;
- exact versions are logged with every run.

Any remaining difference is reported as a threat to validity.

### Quality evaluation

All images are generated on the RTX 5080, which supports every configuration. Quality is a property of the
configuration, not the device. A subset is regenerated on the T4 and GH200 to confirm that kernel differences
don't change it.

1. **Pilot:** all 48 configurations × 2,000 images. FID via `clean-fid` with fixed ImageNet statistics, plus improved precision/recall.
2. **Promote:** keep configurations on or near the preliminary Pareto frontier.
3. **Confirm:** 10,000 images × 3 seeds.

FID is compared only at equal sample counts, because it is biased at small N.

### Analysis

- **H1/H2:** per device and batch size, regress latency and energy on NFE and on FLOPs, and report the error by
  acceleration type. Build FID–latency and FID–energy frontiers. Compute Kendall's τ between devices for plain
  and accelerated configurations, with bootstrap intervals. List every configuration that is on the frontier
  on one device but dominated on another.
- **H3:** compare the FID increase caused by caching across model family and step count. As a mechanism check,
  relate it to the cosine similarity of consecutive-step block outputs.

## Repository structure

```
pareto-shift/
├── README.md
├── docs/
│   ├── proposal/              # Current proposal: LaTeX source and PDF
│   └── ...                    # Earlier drafts, presentation, talking points
├── configs/                   # Sweep manifest: models, samplers, steps, accelerations, seeds
│   └── devices/               # Per-device settings (rtx5080, colab-t4, ikaros-gh200)
├── src/pareto_shift/
│   ├── models/                # Loading the public DiT and SiT checkpoints
│   ├── samplers/              # DDIM, DPM-Solver (DiT); Euler, Heun ODE (SiT); NFE and FLOP counting
│   ├── benchmark/             # Timing harness: warm-up, CUDA events, NVML memory and energy
│   ├── evaluation/            # clean-fid FID, improved precision/recall
│   └── analysis/              # NFE/FLOP regression, Pareto frontiers, Kendall's τ, plots
├── scripts/                   # Entry points: timing sweeps, sample generation, metrics, analysis
├── cluster/ikaros/            # Job scripts for the Ikaros GH200 nodes
├── notebooks/                 # Colab T4 timing notebook, exploratory analysis
├── results/
│   ├── raw/                   # Raw timing and energy logs (not committed)
│   ├── samples/               # Generated images (not committed)
│   ├── metrics/               # Aggregated FID, precision/recall, latency and energy tables
│   └── figures/               # Pareto plots and rank-agreement figures
└── tests/
```

Caching and quantization will live in `src/pareto_shift/` once implemented.

## Schedule (13 weeks)

| Weeks | Phase | Milestone |
|---|---|---|
| 1–2 | **Setup:** checkpoints, pinned environment on all three GPUs, protocol, kernel smoke test | Kernel availability fixed |
| 3–4 | **Harness:** timing, energy and FLOP counting on the RTX 5080; caching and INT8/FP8 validated against fp16 | Harness validated |
| 5–7 | **Pilot:** 2k images, FID, precision/recall, preliminary frontiers | Pilot frontier |
| 8–9 | **Cross-device:** T4 and GH200 timing sweeps, output checks | Cross-device data |
| 10–11 | **Confirm:** 10k images × 3 seeds | |
| 12–13 | **Analyze:** regressions, frontiers, τ, report | |

About 6 GPU-hours on the RTX 5080 cover the pilot; the confirmation stage adds about a day. If compute is
short, cut in this order: (1) FP8, (2) batch 16 on Ikaros, (3) the 32-step level for accelerated
configurations. Both model families, caching and the cross-device timing are never cut.

## Deliverables

1. An open benchmark harness and configuration manifest.
2. Per-device Pareto frontiers and rank-agreement statistics.
3. Evidence on when NFE or FLOPs can stand in for measured cost, and when an efficiency claim must state its hardware.

**Out of scope:** serving systems, training, and exhaustive solver searches.

## Documents

- [Project proposal (PDF)](docs/proposal/proposal.pdf) · [LaTeX source](docs/proposal/proposal.tex)
- [Proposal presentation (PPTX)](docs/Hardware_Efficient_GenModels_Proposal.pptx)
- [Presentation talking points and Q&A prep (PDF)](docs/Talking_Points_QA_Prep.pdf)
- Earlier proposal drafts:
  [submitted version (PDF)](docs/proposal_el23408.pdf),
  [two-page draft (PDF)](docs/Advanced_AI_Research_Proposal_Two_Pages.pdf),
  [first draft (PDF)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.pdf>),
  [first draft (DOCX)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.docx>)

## References

1. N. Ma et al. "SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers." ECCV, 2024.
2. Y. Lipman et al. "Flow Matching for Generative Modeling." ICLR, 2023.
3. J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
4. W. Peebles, S. Xie. "Scalable Diffusion Models with Transformers." ICCV, 2023.
5. J. Song, C. Meng, S. Ermon. "Denoising Diffusion Implicit Models." ICLR, 2021.
6. C. Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." NeurIPS, 2022.
7. X. Ma, G. Fang, X. Wang. "DeepCache: Accelerating Diffusion Models for Free." CVPR, 2024.
8. P. Selvaraju et al. "FORA: Fast-Forward Caching in Diffusion Transformer Acceleration." arXiv:2407.01425, 2024.
9. J. Wu et al. "PTQ4DiT: Post-training Quantization for Diffusion Transformers." NeurIPS, 2024.
10. X. Yin et al. "AccelAes: Accelerating Diffusion Transformers for Training-Free Aesthetic-Enhanced Image Generation." ECCV, 2026.
11. A. Aggarwal, A. Shrivastava, M. Gwilliam. "Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model." ICLR, 2026.
12. J. Liu et al. "A Survey on Cache Methods in Diffusion Models: Toward Efficient Multi-Modal Generation." arXiv:2510.19755, 2025.
13. M. Dehghani et al. "The Efficiency Misnomer." ICLR, 2022.
14. J.-W. Chung et al. "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization." NeurIPS Datasets and Benchmarks, 2025.
15. G. Parmar, R. Zhang, J.-Y. Zhu. "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." CVPR, 2022.
16. T. Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS, 2019.
17. M. J. Chong, D. Forsyth. "Effectively Unbiased FID and Inception Score and Where to Find Them." CVPR, 2020.
