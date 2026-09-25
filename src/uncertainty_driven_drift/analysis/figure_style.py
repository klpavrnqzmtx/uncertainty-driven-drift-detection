"""Shared house style for the stream figures: palette, typography, naming.

Three things live here so that every panel agrees on them.

**Palette.**  Four series colours, picked by name (green/red/purple/blue) rather
than the earlier Okabe-Ito-derived set. Trade-off, stated plainly: green vs. red
(epistemic vs. input) is the single worst pairing under red-green colour-vision
deficiency (deuteranopia/protanopia, ~8% of men) — the two hues can become
indistinguishable. Identity does not rest on colour alone, though: every signal
also has its own marker shape (circle/square/triangle/diamond, see
``SIGNAL_MARKER``) and every row carries its own text label, so the figure still
disambiguates under CVD or greyscale printing — just with a weaker colour cue
than the previous palette provided.

**Typography.**  One place to change sizes, so a figure is never bold-titled in
one panel and 5pt-italic in another.  ``pdf.fonttype=42`` embeds TrueType so the
PDF is editable and searchable in the paper.

**Naming.**  Figures showed internal registry ids — ``adwin_epistemic``,
``ph_total``, ``audio_kn_esc50``, ``gauss`` — which are meaningless to a reader
and inconsistent with the prose.  The maps below are the single source of the
names a reader sees.
"""

from __future__ import annotations

from typing import Any, Dict

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------

PALETTE: Dict[str, str] = {
    "epistemic": "#2CA02C",    # green — ours
    "total": "#9467BD",        # purple — the UDD baseline
    "input": "#D62728",        # red — model-free baseline
    "supervised": "#0072B2",   # blue — labelled reference, not a rival
}

INK = "#1a1a1a"        # primary text
INK_MUTED = "#6b6b6b"  # secondary text (phase tags, captions)
RULE = "#c8c8c8"       # hairline grid / axis
BAND = "#f2f2f2"       # alternating phase shading
NOVEL_INK = "#8f2d00"  # phase tags after the boundary


def signal_of(detector: str) -> str:
    """Which signal a detector reads, from its registry name."""
    n = detector.lower()
    if "epistemic" in n or "_mi" in n:
        return "epistemic"
    if "total" in n or "entropy" in n:
        return "total"
    if "input" in n:
        return "input"
    if n.startswith(("ddm", "eddm", "pilot_ddm", "pilot_eddm")):
        return "supervised"
    return "supervised"


def signal_color(detector: str) -> str:
    return PALETTE[signal_of(detector)]


#: Marker per signal — a filled shape, so colour is never the only cue: the
#: figure survives greyscale printing and the CVD floor on the input hue.
SIGNAL_MARKER = {
    "epistemic": "o",
    "total": "s",
    "input": "^",
    "supervised": "D",
}


def signal_marker(detector: str) -> str:
    return SIGNAL_MARKER[signal_of(detector)]


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

ALGO_LABEL = {
    "adwin": "ADWIN",
    "kswin": "KSWIN",
    "ph": "Page-Hinkley",
    "page_hinkley": "Page-Hinkley",
    "ddm": "DDM",
    "eddm": "EDDM",
    "chi2": "chi-square",
}

SIGNAL_LABEL = {
    "epistemic": "Epistemic",
    "total": "Total entropy",
    "input": "Input stat.",
    "supervised": "Error-based signal",
}


def detector_label(detector: str) -> str:
    """Registry name -> the label a reader sees, e.g. ``Epistemic · ADWIN``.

    Spelled out per row rather than relying on a colour key: a row label is text,
    so identity survives greyscale printing and colour-vision deficiency.
    """
    n = detector.lower()
    sig = signal_of(n)
    if sig == "supervised":
        return f"{SIGNAL_LABEL[sig]} (labels)"
    algo = next((ALGO_LABEL[k] for k in ("adwin", "kswin", "page_hinkley", "ph")
                 if n.startswith(k)), n.split("_")[0].upper())
    return f"{SIGNAL_LABEL[sig]} · {algo}"


#: Stream channel ids (short or long form) -> the name used in the paper.
CHANNEL_LABEL = {
    # audio — known
    "gauss": "Gaussian", "gaussian_noise": "Gaussian",
    "pink": "Pink", "pink_noise": "Pink",
    "hum": "Mains hum", "hum_noise": "Mains hum",
    "lowpass": "Low-pass", "highpass": "High-pass",
    "notch": "Band-stop", "band_stop": "Band-stop",
    "quant": "Quantise", "quantization": "Quantise",
    "reverb": "Reverb", "echo": "Echo",
    "clip": "Clipping", "clipping": "Clipping", "gain": "Gain",
    # audio — novel. "Impulse noise" rather than "Impulse clicks": the image
    # streams use the same short name `impulse` for salt-and-pepper pixel noise,
    # and a shared map must not label an image corruption in audio terms.
    "impulse": "Impulse noise", "impulse_noise": "Impulse noise",
    "dropout": "Packet loss", "packet_loss": "Packet loss",
    "pitch": "Pitch shift", "pitch_shift": "Pitch shift",
    "speed": "Speed", "time_stretch": "Time stretch",
    "clean": "Clean", "identity": "Clean",
    # image corruption families (CIFAR-10-C / MNIST-C short names)
    "bright": "Brightness", "brightness": "Brightness",
    "contrast": "Contrast", "jpeg": "JPEG", "jpeg_compression": "JPEG",
    "shot": "Shot noise", "shot_noise": "Shot noise",
    "defocus": "Defocus blur", "defocus_blur": "Defocus blur",
    "glass": "Glass blur", "glass_blur": "Glass blur",
    "motion": "Motion blur", "motion_blur": "Motion blur",
    "zoom": "Zoom blur", "zoom_blur": "Zoom blur",
    "snow": "Snow", "frost": "Frost", "fog": "Fog",
    "elastic": "Elastic", "elastic_transform": "Elastic",
    "pixel": "Pixelate", "pixelate": "Pixelate",
    "rotate": "Rotation", "translate": "Translation",
    "scale": "Scale", "shear": "Shear", "stripe": "Stripe",
    "dotted_line": "Dotted line", "spatter": "Spatter",
    "canny_edges": "Canny edges", "zigzag": "Zigzag",
    # wireless: QuaDRiGa's vehicle-obstructed highway scenario is named NLOSv;
    # the paper calls both obstructed conditions NLOS, so the figure does too.
    "Highway NLOSv": "Highway NLOS", "highway_nlosv": "Highway NLOS",
}


def channel_label(name: Any) -> str:
    key = str(name)
    return CHANNEL_LABEL.get(key, CHANNEL_LABEL.get(key.lower(), key.replace("_", " ")))


_DATASET_TITLE = {
    "speech_commands": "Speech Commands v2",
    "esc50": "ESC-50",
    "urbansound8k": "UrbanSound8K",
}

#: Image/wireless streams do not carry a `dataset` extra, so they are named from
#: the stream's own registry name instead.
_STREAM_TITLE = {
    "cifar_known_novel": "CIFAR-10",
    "cifar100_known_novel": "CIFAR-100",
    "mnist_known_novel": "MNIST-C",
    "mnist_c": "MNIST-C",
    "fashion_known_novel": "Fashion-MNIST",
    "kmnist_known_novel": "KMNIST",
    "camelyon17_known_novel": "Camelyon17",
    "cifar_known_ambiguous": "CIFAR-10 (ambiguous)",
    "emnist_label_prior": "EMNIST (label prior)",
    "quadriga_known_novel": "QuaDRiGa",
}


def arm_title(metrics: Dict[str, Any]) -> str:
    """``Speech Commands v2 · ResNet-18 11.2 M`` from a run's metrics.json.

    Experiment ids like ``audio_kn_esc50_resnet18`` are for directories, not for
    a reader; returns an empty string for non-audio runs so their panels keep
    their own captions.
    """
    exp = str(metrics.get("experiment", ""))
    stream = metrics.get("stream") or {}
    extras = stream.get("extras") or {}
    dataset = extras.get("dataset")
    if not dataset or dataset not in _DATASET_TITLE:
        # Fall back to the stream name for the image and wireless arms.
        name = _STREAM_TITLE.get(str(stream.get("name", "")))
        if not name:
            return ""
        if "vit" in exp and "laplace" in exp:
            return f"{name} · ViT, last-layer Laplace"
        if "vit" in exp:
            return f"{name} · ViT, MC-Dropout"
        if "laplace" in exp:
            return f"{name} · ResNet-20, last-layer Laplace"
        return f"{name} · ResNet-20, MC-Dropout"
    if exp.endswith("_resnet18"):
        backbone = "ResNet-18 11.2 M"
    elif exp.endswith("_wide"):
        backbone = "CNN-wide 1.58 M"
    elif exp.endswith("_laplace"):
        backbone = "CNN 247 K, Laplace"
    else:
        backbone = "CNN 247 K"
    return f"{_DATASET_TITLE[dataset]} · {backbone}"


# --------------------------------------------------------------------------
# typography
# --------------------------------------------------------------------------

def rc(base: float = 7.0) -> Dict[str, Any]:
    """rcParams for a conference-width panel. ``base`` is the body size in pt."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 1.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        # Tick labels in full ink: the muted grey was hard to read at paper size.
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": base - 1,
        "legend.frameon": False,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "lines.linewidth": 1.2,
        "text.color": INK,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,      # embed TrueType: editable/searchable in the paper
        "ps.fonttype": 42,
    }
