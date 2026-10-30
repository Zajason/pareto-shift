# pareto-shift

**Do efficient generative sampling choices transfer across GPUs?**
Training-free acceleration of diffusion and flow-matching transformers.

Project for *Advanced Topics in Artificial Intelligence* (NTUA ECE, 2026–27). Author: Iason-Daniil Zakynthinos.

---

## Overview

Sampling speedups for image generators are usually measured on one GPU. Solvers reduce the number of network
evaluations, **feature caching** reduces the computation inside an evaluation, and **INT8/FP8 quantization**
changes its arithmetic cost, so these methods change different parts of inference cost and can behave
differently on different hardware.

This project tests whether the preferred sampling configuration changes across GPU platforms **at fixed image
quality**, and measures what it costs to transfer a choice from one device to another. It uses pretrained
**DiT** (diffusion) and **SiT** (flow matching) transformers and compares plain sampling, feature caching and
INT8/FP8 inference on three GPUs. No model is trained.

**Research question.** Do quality–efficiency conclusions about diffusion and flow-matching sampling remain
valid across GPU platforms when training-free accelerations are introduced?

**Primary outcome: transfer penalty.** For a pair of GPUs and an FID ceiling, take the fastest eligible
configuration on the source GPU and measure it on the target:

> transfer penalty = target latency of the source's choice ÷ fastest eligible target latency − 1

A penalty of at most 5% counts as near-optimal. Kernels a GPU lacks are reported separately and excluded from
transfer comparisons.

| | Hypothesis |
|---|---|
| **H1 — Transfer baseline** | Within each batch size, one multiplicative latency factor per device describes plain sampling across platforms, so the best choice transfers. Configuration-by-device interactions are the competing explanation. |
| **H2 — Cost explanation** | Caching and low-precision execution add interactions that NFE or operation counts can't explain. A small model of arithmetic throughput, memory traffic and fixed overhead should better predict held-out latency and transferred choices. |
| **H3 — Quality mechanism (exploratory)** | Relative feature-reuse error predicts caching-induced quality loss better than step count alone, including across models. No assumption that flow matching tolerates caching better: straighter paths don't guarantee slowly changing hidden features. |

Stable choices with tightly bounded penalties are also an informative result.

## Experimental design

**2 × 2 × 3 × 4 = 48** configurations before availability exclusions:

| Factor | Diffusion · DiT-XL/2 | Flow · SiT-XL/2 | Levels |
|---|---|---|---|
| Checkpoint | ImageNet-256, public | ImageNet-256, public | 2 |
| Sampler | DDIM · DPM-Solver (specified variant) | Euler · Heun (ODE) | 2 each |
| Steps | 8 · 16 · 32 | 8 · 16 · 32 | 3 |
| Execution mode | FP16 · FORA-style caching · INT8 W8A8 · FP8 W8A8 | same | 4 |
| Batch size | 1 · 16 (timing conditions) | 1 · 16 (timing conditions) | — |

Actual NFE, including solver substages, guidance and cached calls, is logged separately from nominal steps.

### Hardware

| Platform | Notes |
|---|---|
| NVIDIA RTX 5080 (local) | Blackwell. Most quality evaluation runs here. |
| Google Colab T4 | Turing. No bf16 or FP8. |
| GH200 on the Ikaros cluster | Hopper. Limited hours. |

These are three tested platforms, not samples of a GPU generation. End-to-end smoke tests establish which
kernels each platform supports before the sweep.

### Implementation and manifest

- The manifest fixes checkpoint, timestep grid, solver order, guidance, labels, seeds, compilation policy and VAE.
- Quantization specifies calibration data, scale granularity, excluded layers, accumulation precision and the
  actual low-precision kernels. Calibration and evaluation seeds are disjoint. Fake quantization is not counted
  as accelerated execution.
- Caching fixes the cached attention/MLP outputs, refresh interval, initial refresh and solver-substage
  behaviour before evaluation.
- PyTorch and torchao versions are pinned where possible; driver, CUDA, host and remaining stack differences
  are recorded.

### Quality

1. **Screen:** 2,000 images per configuration. FID via `clean-fid` with fixed ImageNet reference statistics, plus precision/recall.
2. **Freeze:** three FID ceilings spanning the useful pilot range, and the selection rules, before confirmation.
3. **Confirm:** three independent 10,000-image runs per retained configuration. Per-run scores and uncertainty are reported, not pooled.

A prespecified subset (both models, every shared execution mode, near-threshold cases) is regenerated on the
other GPUs before RTX 5080 scores are reused there. If equivalence within a declared tolerance is unresolved,
those candidates are evaluated on the target directly or excluded from transfer claims.

### Timing and energy

- Warm-up and compilation first, then randomized configuration order, repeated across multiple sessions.
- Median and interquartile range for synchronized end-to-end generation; denoising and VAE decoding also timed separately.
- Batch latency, throughput and amortized time per image are kept distinct. Peak memory, power limits and clocks are recorded.
- Energy is secondary: long measurement windows with supported NVML counters, stated power domain and idle treatment.

### Feature-reuse probes (H3)

On a small independent probe set, reused features are compared with full recomputation at the same state and
time: relative L2 error, magnitude change, and timestep/layer summaries. Probes run outside timed workloads.

## Analysis

- **Primary:** transfer penalty for each ordered device pair, batch size and FID ceiling, with timing and
  quality uncertainty propagated. No winner is forced when eligibility or dominance is unresolved.
- **H1/H2:** common-scaling baseline vs configuration-by-device interactions. NFE-only and operation-count
  baselines, then a hardware-aware cost model (compute, memory, overhead) with rates from independent kernel
  probes. Validated on held-out configuration groups and, exploratorily, one held-out device at a time.
- **H3:** whether feature-reuse diagnostics predict the quality change beyond model, solver and step count;
  cross-model prediction evaluated separately from within-model fit.
- **Secondary:** Pareto frontiers and latency-only Kendall's τ-b on shared configurations.

## Repository structure

```
pareto-shift/
├── README.md
├── docs/
│   ├── proposal/              # Current proposal: LaTeX source and PDF, plus the ReportLab revision
│   └── ...                    # Earlier drafts, presentation, talking points
├── configs/                   # Manifest: models, samplers, steps, execution modes, seeds, compile policy
│   └── devices/               # Per-device settings (rtx5080, colab-t4, ikaros-gh200)
├── src/pareto_shift/
│   ├── models/                # Loading the public DiT and SiT checkpoints
│   ├── samplers/              # DDIM, DPM-Solver (DiT); Euler, Heun ODE (SiT); NFE logging
│   ├── benchmark/             # Timing harness: warm-up, CUDA events, NVML memory and energy
│   ├── evaluation/            # clean-fid FID, precision/recall
│   └── analysis/              # Transfer penalty, cost models, Pareto frontiers, Kendall's τ, plots
├── scripts/                   # Entry points: timing sweeps, sample generation, metrics, analysis
├── cluster/ikaros/            # Job scripts for the Ikaros GH200 nodes
├── notebooks/                 # Colab T4 timing notebook, exploratory analysis
├── results/
│   ├── raw/                   # Raw timing and energy logs (not committed)
│   ├── samples/               # Generated images (not committed)
│   ├── metrics/               # Aggregated quality, latency and energy tables
│   └── figures/               # Transfer-penalty, Pareto and rank-agreement figures
└── tests/
```

Caching, quantization and the feature-reuse probes will live in `src/pareto_shift/` once implemented.

## Schedule (13 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Checkpoints, environments, complete kernel smoke tests, manifest |
| 3–4 | Solver/caching validation, quantization calibration, timing protocol |
| 5–7 | Quality pilot, feature probes, frozen thresholds and shortlist |
| 8–9 | Cross-platform timing, kernel probes, quality checks |
| 10–11 | Independent quality confirmation, targeted target-device evaluation |
| 12–13 | Held-out analysis, uncertainty, plots, report |

**Feasibility.** Nothing is trained. The 2,000-image pilot over all 48 configurations needs roughly 6 GPU-hours
on the RTX 5080 and the confirmation stage about a day; the exact budget is set from measured pilot throughput.
The T4 and GH200 only run timing sweeps, kernel probes and the quality re-check subset. If time is short, cut
in this order: (1) held-out-device validation, (2) FP8, (3) batch 16 on the GH200, (4) the 32-step level for
accelerated configurations. The transfer-penalty analysis on shared configurations is never cut.

## Deliverables

1. Evidence on whether the preferred model–sampler–acceleration configuration changes across platforms at fixed image quality.
2. Transfer penalties with uncertainty, and an execution-cost explanation of the changes.
3. An open benchmark harness and manifest.

Claims are limited to the tested platforms and configuration space. **Out of scope:** serving, training, and
exhaustive solver or cache-policy optimization.

## Documents

- [Project proposal (PDF)](docs/proposal/proposal.pdf) · [LaTeX source](docs/proposal/proposal.tex)
- [Revised proposal, ReportLab version (PDF)](docs/proposal/revised-proposal.pdf): the revision the current LaTeX version is based on
- [Proposal presentation, 3 slides (PPTX)](docs/Hardware_Efficient_GenModels_Proposal.pptx)
- [Presentation talking points and Q&A prep (PDF)](docs/Talking_Points_QA_Prep.pdf) (describes an earlier design)
- Earlier proposal drafts:
  [submitted version (PDF)](docs/proposal_el23408.pdf),
  [two-page draft (PDF)](docs/Advanced_AI_Research_Proposal_Two_Pages.pdf),
  [first draft (PDF)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.pdf>),
  [first draft (DOCX)](<docs/Research Proposal Does Hardware Change Which Generative Model Is Efficient.docx>)

## References

1. J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
2. Y. Lipman et al. "Flow Matching for Generative Modeling." ICLR, 2023.
3. W. Peebles, S. Xie. "Scalable Diffusion Models with Transformers." ICCV, 2023.
4. N. Ma et al. "SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers." ECCV, 2024.
5. J. Song, C. Meng, S. Ermon. "Denoising Diffusion Implicit Models." ICLR, 2021.
6. C. Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." NeurIPS, 2022.
7. P. Selvaraju et al. "FORA: Fast-Forward Caching in Diffusion Transformer Acceleration." arXiv:2407.01425, 2024.
8. J. Wu et al. "PTQ4DiT: Post-training Quantization for Diffusion Transformers." NeurIPS, 2024.
9. X. Yin et al. "AccelAes: Accelerating Diffusion Transformers for Training-Free Aesthetic-Enhanced Image Generation." ECCV, 2026.
10. A. Aggarwal, A. Shrivastava, M. Gwilliam. "Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model." ICLR, 2026.
11. M. Dehghani et al. "The Efficiency Misnomer." ICLR, 2022.
12. J.-W. Chung et al. "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization." NeurIPS Datasets and Benchmarks, 2025.
13. G. Parmar, R. Zhang, J.-Y. Zhu. "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." CVPR, 2022.
14. T. Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS, 2019.
15. M. J. Chong, D. Forsyth. "Effectively Unbiased FID and Inception Score and Where to Find Them." CVPR, 2020.
