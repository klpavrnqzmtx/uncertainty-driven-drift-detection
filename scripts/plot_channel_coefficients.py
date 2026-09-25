"""Plot Quadriga SISO channel coefficients for all four scenarios.

Saves:  results/figures/wireless_kn_comparison/channel_coefficients.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


DATA_ROOT = Path("./database/SISO_channel_quadriga")
OUT_PATH  = Path("./results/figures/wireless_kn_comparison/channel_coefficients.png")

CHANNELS = [
    ("channel_01", "Highway LOS",   "#1f77b4", "known"),
    ("channel_02", "Highway NLOSv", "#aec7e8", "known"),
    ("channel_03", "Urban LOS",     "#ff7f0e", "novel"),
    ("channel_04", "Urban NLOS",    "#d62728", "novel"),
]


def load_h(prefix: str) -> np.ndarray:
    files = sorted(DATA_ROOT.glob(f"{prefix}_*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat file matching {prefix}_*.mat in {DATA_ROOT}")
    mat = scipy.io.loadmat(str(files[0]))
    h = mat["channel_data"]["h"][0, 0][:, 0]
    return h.astype(np.complex128)


def main() -> None:
    hs = [(label, color, phase, load_h(ch)) for ch, label, color, phase in CHANNELS]

    # Reference RMS from known channels only
    ref_rms = float(np.mean([np.sqrt(np.mean(np.abs(h) ** 2)) for _, _, phase, h in hs if phase == "known"]))

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), dpi=150, sharex=True)
    fig.suptitle(
        "Quadriga SISO Channel Coefficients — |h[t]| per scenario\n"
        f"(normalised by known-channel reference RMS = {ref_rms:.3e})",
        fontsize=12, fontweight="bold",
    )

    t = np.arange(5000)

    for ax, (label, color, phase, h) in zip(axes, hs):
        raw_rms   = float(np.sqrt(np.mean(np.abs(h) ** 2)))
        h_norm    = h / ref_rms
        norm_rms  = raw_rms / ref_rms
        eff_snr_db = 10 * np.log10(norm_rms ** 2 * 10 ** (28.0 / 10))

        # Raw magnitude (faint) + smoothed envelope
        mag = np.abs(h_norm)
        window = 50
        envelope = np.convolve(mag, np.ones(window) / window, mode="same")

        ax.fill_between(t, 0, mag, color=color, alpha=0.18, linewidth=0)
        ax.plot(t, mag,      color=color, lw=0.4, alpha=0.55, label="_nolegend_")
        ax.plot(t, envelope, color=color, lw=1.6, label=f"smoothed envelope (w={window})")

        ax.axhline(norm_rms, color="black", ls="--", lw=1.0, alpha=0.7,
                   label=f"RMS = {norm_rms:.3f}  (eff. SNR @ 28 dB nom. = {eff_snr_db:.1f} dB)")

        phase_tag = "KNOWN" if phase == "known" else "NOVEL"
        ax.set_title(f"{label}  [{phase_tag}]", fontsize=10, fontweight="bold",
                     color=color, loc="left")
        ax.set_ylabel("|h_norm[t]|")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Symbol index t")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    # Summary table
    print("\nChannel summary (nominal SNR = 28 dB):")
    print(f"{'Channel':22s}  {'Phase':6s}  {'raw RMS':10s}  {'norm RMS':10s}  {'eff SNR':8s}")
    print("-" * 65)
    for label, color, phase, h in hs:
        raw_rms  = float(np.sqrt(np.mean(np.abs(h) ** 2)))
        norm_rms = raw_rms / ref_rms
        eff_db   = 10 * np.log10(norm_rms ** 2 * 10 ** (28.0 / 10))
        print(f"{label:22s}  {phase:6s}  {raw_rms:.4e}    {norm_rms:.4f}      {eff_db:5.1f} dB")


if __name__ == "__main__":
    main()
