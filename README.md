# Uncertainty-Driven Unsupervised Detection of Reliability-Relevant Distribution Drift

Code for the paper of the same name.

A frozen classifier is streamed through a **known** phase (shifts it saw during training) followed
by a **novel** phase (shifts it never saw). Every batch yields three label-free signals: epistemic
uncertainty, total predictive entropy, and an input statistic. Each signal feeds standard
sequential drift detectors (ADWIN, KSWIN, Page-Hinkley). An error-rate detector (DDM), which needs
deployment labels, runs alongside as a supervised reference.

## Layout

| path | contents |
|---|---|
| `src/uncertainty_driven_drift/` | streams and corruptions, models (MC dropout, last-layer Laplace), uncertainty decomposition, detectors, runner, plotting |
| `configs/experiments/` | one YAML file per experiment arm |
| `scripts/` | backbone training, threshold calibration, tables, supplementary analyses, figures |
| `tests/` | unit tests |

## Install

Python 3.10 or newer:

```bash
pip install -e '.[torch,drift,plot,audio,dev]'
pytest -q
```

Training needs a GPU with compute capability 7.5 or newer (Turing onward), because current PyTorch
wheels ship no Pascal or Volta kernels. `UB_ALLOW_CPU=1` runs on the CPU instead, which is valid but
much slower. The detectors come from `river`, whose alarm times can change between releases; every
run records its package versions in `metrics.json`.

## Data

- **MNIST, EMNIST, CIFAR-10, CIFAR-100:** torchvision downloads them into `artifacts/` on first use.
- **Speech Commands v2, ESC-50, UrbanSound8K:**
  ```bash
  bash scripts/download_audio.sh                  # about 9 GB
  python scripts/prepare_audio.py --dataset all   # decode into waveform caches
  ```
- **ImageNet:** gated on the Hugging Face Hub (`ILSVRC/imagenet-1k`). With `HF_TOKEN` set:
  ```bash
  python scripts/hf_imagenet_extract.py --out artifacts/imagenet/val --split val                          # the stream
  python scripts/hf_imagenet_extract.py --out artifacts/imagenet/train_slice --split train --n-shards 20   # fine-tuning only
  ```
- **ViT-B/16 weights:** `timm` downloads `vit_base_patch16_224.augreg_in21k` (and its ImageNet-1k
  fine-tune, for the ImageNet arms) on first use.
- **Wireless:** QuaDRiGa channel taps, read from `database/QuaDRiGa/<scenario>_taps.mat`. These
  files are not included.

## Run one arm

```bash
uncertainty-driven-drift run --config configs/experiments/cifar_kn_comparison/known_vs_novel_laplace.yaml
```

A run writes `results/raw/<experiment>/<timestamp>/`: per-batch records (`steps.jsonl`), summary
metrics (`metrics.json`) and the resolved config. Backbones train on first use and are cached under
`artifacts/models/`, so this one command runs an arm end to end. `scripts/pretrain_vit.py` and
`scripts/train_wrn_cifar100.py` train the ViT and WRN-28-10 backbones ahead of time.
`configs/experiments/smoke.yaml` is a synthetic stream that finishes in seconds.

| group | configs under `configs/experiments/` |
|---|---|
| MNIST | `mnist_c/known_vs_novel_*.yaml` |
| CIFAR-10 | `cifar_kn_comparison/known_vs_novel*.yaml`; `severity_shift_*.yaml` for the severity stream |
| CIFAR-100 | `cifar100_c/known_vs_novel_{vit,wrn,resnet20}_{mc_dropout,laplace}.yaml` |
| ImageNet | `imagenet_comparison/known_vs_novel_vit_{mc_dropout,laplace}.yaml` |
| Audio | `audio_kn_comparison/<dataset>_known_vs_novel*.yaml` (CNN, CNN-wide and ResNet-18; Speech Commands also has a Laplace arm) |
| Wireless | `wireless_setup/multitap_channel_drift/*.yaml` |
| Benign label-prior shift | `emnist_label_prior/benign_shift.yaml` |

The remaining directories (`mnist_tasks`, `kmnist`, `camelyon17`, `cifar_ambiguity`) hold
additional stream types.

## Threshold calibration

Thresholds are set at a matched false-alarm budget. For every signal and detector, the sweep keeps
the setting with the shortest detection delay among those that alarm on at most 5% of known-phase
batches (or the lowest false-alarm rate, if none does), and writes it into the config:

```bash
python scripts/sweep_detectors.py --config <config> --mode known_novel --select min_delay \
    --seeds 0,1,2,3,4 --target-far 0.05 --apply-config <config> --out results/sweeps/<name>
```

Vision arms are calibrated on seeds 0–4 and audio arms on seeds 7–9.
`APPLY=1 bash scripts/sweep_wireless.sh` calibrates the wireless arms, also on seeds 7–9. The
search grids are defined in `scripts/sweep_detectors.py`.

## Tables and figures

```bash
python scripts/auroc_fpr95_table.py --seeds 0,1,2,3,4 --out results/tables/AUROC_FPR95   # AUROC; FPR at 95/99/100%
python scripts/latex_tables.py auroc  --out results/tables/table1.tex
python scripts/latex_tables.py fpr    --out results/tables/table_fpr.tex
python scripts/streaming_metrics.py --target-far 0.05 --out results/tables/STREAMING_METRICS
python scripts/latex_tables.py stream --out results/tables/table_streaming.tex

python scripts/toy_epistemic_coverage_option_b_compact.py --out results/figures/paper   # Figure 1
python scripts/paper_panel.py --run results/raw/<experiment>/<timestamp> --out results/figures/paper/<name>
PY=python bash scripts/audio_figures.sh                                                 # every audio panel
python scripts/auroc_fpr95_png.py results/tables/AUROC_FPR95.json --out results/figures/paper/auroc_fpr95_table
python scripts/far_targets_table.py results/sweeps/*/selection.json --targets 0,0.01,0.05,0.10 \
    --out results/tables/FAR_TARGETS
python scripts/far_targets_png.py results/tables/FAR_TARGETS.json --far 0.05 \
    --out results/figures/paper/far_targets_summary
```

## Supplementary analyses

```bash
# Laplace posterior diagnostics: calibrated prior precision, posterior locality, runtime per batch
python scripts/laplace_diagnostics.py --config <laplace config> --out results/tables/diagnostics/<name>.json

# Sensitivity to the prior precision, the number of posterior samples S and the batch size B
python scripts/ablate.py --config <config> --param model.params.prior_precision \
    --grid 1,10,100,1000,10000,100000 --fix-prior --seeds 0,1,2 --out results/tables/ablation/<name>.json
python scripts/ablate.py --config <config> --param model.params.n_samples \
    --grid 5,10,20,50,100 --seeds 0,1,2 --out results/tables/ablation/<name>.json
python scripts/ablate.py --config <config> --param dataset.params.batch_size \
    --grid 16,32,64,128,256 --seeds 0,1,2 --out results/tables/ablation/<name>.json

# Embedding-distribution baseline: Frechet distance to a known-phase reference window
python scripts/baseline_driftlens.py --config <config> --seeds 0,1,2,3,4 --out results/tables/baselines/<name>.json
```

The ablations use the CIFAR-10 ResNet-20 Laplace and CIFAR-100 ViT-B/16 Laplace arms.
