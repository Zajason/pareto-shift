# pareto-shift

**Does hardware change which generative model is efficient?**
Evaluating diffusion and flow-matching samplers across heterogeneous GPUs.

Project for *Advanced Topics in Artificial Intelligence* (NTUA ECE, 2026–27). Author: Iason-Daniil Zakynthinos.

---

## Overview

Efficiency claims for image generators are usually made on a single accelerator and summarised by the
number of function evaluations (NFE). This project checks whether those claims hold when the hardware changes.

We compare pretrained **diffusion (DiT)** and **flow-matching (SiT)** models. The two share a nearly identical
transformer backbone, so they form a controlled pair. For each GPU we build quality–cost Pareto frontiers from
FID, precision/recall, latency, peak memory and energy per image. No model is trained.

**Research question.** Do diffusion and flow-matching sampler rankings remain stable across GPU classes and
numerical precisions?

| | Hypothesis |
|---|---|
| **H1 — NFE ≠ cost** | NFE tracks latency within one GPU and sampler, but misses solver overhead, memory bandwidth and precision support. |
| **H2 — Rankings move** | Near-frontier configurations can swap rank between devices, so a Pareto front built on one GPU can mislead. |
| **H3 — Straight paths pay off** | Flow matching loses quality more slowly in the low-step regime, compared at matched *measured* cost. |

Either outcome is useful. Stable rankings support NFE as a proxy. Unstable rankings mean efficiency claims
need to state their hardware.

## Experimental design

A full factorial of **2 × 2 × 3 × 2 = 24** model–sampler–step–precision configurations:

| Factor | Diffusion · DiT | Flow · SiT | Levels |
|---|---|---|---|
| Checkpoint | ImageNet-256, public | ImageNet-256, public | 2 |
| Sampler | DDIM · DPM-Solver | Euler · Heun (ODE) | 2 each |
| Steps | 8 · 16 · 32 | 8 · 16 · 32 | 3 |
| Precision | bf16 · fp32 | bf16 · fp32 | 2 |
| Batch size | 1 · 8 (timing only) | 1 · 8 (timing only) | — |

Fixed throughout: classifier-free guidance scale, compile settings, seeds and class labels. Every network call
counts toward NFE.

### Hardware

| Device | Role |
|---|---|
| Local GPU | Workhorse: quality generation, pilot sweeps |
| Google Colab T4 (Turing, no native bf16) | Timing sweeps only |
| Ikaros GH200 (Hopper-class, limited hours) | Short timing runs and validation |

Each run uses one GPU, so results stay comparable.

### Measurement protocol (identical on every device)

1. **Pin** software versions, seeds and class labels.
2. **Warm up** kernels before any timing.
3. **Time** with repeated CUDA-event runs.
4. **Measure** peak memory and energy through NVML.
5. **Report** median and interquartile range.

### Quality evaluation

Images are generated once per configuration, not once per GPU. Same weights and seeds give the same images up
to numerical noise, and selected outputs are re-generated on every device to confirm this.

1. **Pilot:** all 24 configurations × 2,000 images. FID via `clean-fid` with fixed ImageNet reference statistics.
2. **Promote:** keep the configurations on or near the preliminary Pareto frontier.
3. **Confirm:** 10,000 images × 3 seeds. FID and improved precision/recall.

### Analysis

- **H1:** regress latency and energy on NFE per device, then inspect the sampler-specific residuals.
- **H2:** build FID–latency and FID–energy Pareto frontiers per device and flag configurations whose status
  changes. Compare device rankings with Kendall's τ.
- **H3:** compare the FID increase at matched measured cost, not at matched NFE.

## Repository structure

```
pareto-shift/
├── README.md
├── docs/                      # Proposal, earlier drafts, presentation and talking points
├── configs/                   # Sweep manifest: models, samplers, steps, precisions, seeds
│   └── devices/               # Per-device settings (local, colab-t4, ikaros-gh200)
├── src/pareto_shift/
│   ├── models/                # Loading the public DiT and SiT checkpoints
│   ├── samplers/              # DDIM, DPM-Solver (DiT); Euler, Heun ODE (SiT); NFE counting
│   ├── benchmark/             # Timing harness: warm-up, CUDA events, NVML memory and energy
│   ├── evaluation/            # clean-fid FID, improved precision/recall
│   └── analysis/              # NFE regression, Pareto frontiers, Kendall's τ, plots
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

## Schedule (13 weeks)

| Weeks | Phase | Milestone |
|---|---|---|
| 1–2 | **Setup:** checkpoints, environment, protocol | |
| 3–4 | **Harness:** timing, memory, energy | Harness validated |
| 5–6 | **Pilot:** 2k images, FID, preliminary frontier | Pilot frontier |
| 7–8 | **Cross-device:** T4 and GH200 sweeps | Cross-device data |
| 9–10 | **Confirm:** 10k images × 3 seeds | |
| 11–13 | **Analyze:** NFE, Pareto, τ, report | |

If compute gets tight, cut in this order: (1) drop fp32 quality generation, (2) run only batch size 1 on Ikaros,
(3) if a checkpoint does not fit, run a small parameter-efficient pilot instead of full training. Both model
families, all three step counts and the cross-device timing are always kept.

## Deliverables

1. A reproducible benchmark script and configuration manifest.
2. Hardware-specific Pareto plots and rank-agreement statistics.
3. Evidence-based guidance on when NFE is a trustworthy efficiency proxy.

**Out of scope:** serving-system design, training from scratch, and exhaustive solver searches.

## Documents

- [Project proposal (PDF)](docs/proposal_el23408.pdf)
- [Proposal presentation (PPTX)](docs/Hardware_Efficient_GenModels_Proposal.pptx)
- [Presentation talking points and Q&A prep (PDF)](docs/Talking_Points_QA_Prep.pdf)
- Earlier proposal drafts:
  [two-page draft (PDF)](docs/Advanced_AI_Research_Proposal_Two_Pages.pdf),
  [first draft (PDF)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.pdf>),
  [first draft (DOCX)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.docx>)

## References

1. M. Dehghani et al. "The Efficiency Misnomer." ICLR, 2022.
2. J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
3. T. Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS, 2019.
4. Y. Lipman et al. "Flow Matching for Generative Modeling." ICLR, 2023.
5. C. Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." NeurIPS, 2022.
6. N. Ma et al. "SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers." ECCV, 2024.
7. G. Parmar, R. Zhang, J.-Y. Zhu. "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." CVPR, 2022.
8. W. Peebles, S. Xie. "Scalable Diffusion Models with Transformers." ICCV, 2023.
9. J. Song, C. Meng, S. Ermon. "Denoising Diffusion Implicit Models." ICLR, 2021.
