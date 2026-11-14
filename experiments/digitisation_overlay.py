"""Overlay the digitised values on the paper's own figure images, using the
digitiser's own axis calibration, to check tools/digitise_figures.py.

    pdfimages -png paper.pdf figs/f      (once)
    python experiments/digitisation_overlay.py figs docs/investigation

Writes one PNG per figure. White-filled crosses are the digitised points; a
point off its line is a digitisation error.
"""
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import digitise_figures as dg                                     # noqa: E402
from slidingcurve import markets                                  # noqa: E402

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402


def axis_maps(img, x_ticks, y_ticks, x_limits=None):
    left, right, top, bottom = dg.frame(img)
    rows = dg.grid_rows(img, left, right, top, bottom)
    ys, yi = np.polyfit(rows[::-1], y_ticks, 1)
    if x_limits is None:
        cols = dg.grid_cols(img, left, right, top, bottom)
        xs, xi = np.polyfit(cols, x_ticks, 1)
    else:
        xs = (x_limits[1] - x_limits[0]) / (right - left)
        xi = x_limits[0] - xs * left
    return (lambda x: (np.asarray(x) - xi) / xs), (lambda y: (np.asarray(y, dtype=float) - yi) / ys)


def overlay(ax, img, px, py, xs, series, title):
    ax.imshow(img)
    for name, vals in series.items():
        v = np.array([np.nan if a is None else a for a in vals], dtype=float)
        ax.plot(px(xs), py(v), "x", ms=6, mew=1.6, color="k")
        ax.plot(px(xs), py(v), "x", ms=4, mew=0.8, color="w")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main(src, dst):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    figs = markets.paper_figures()

    # Figure 4: four panels
    img = np.asarray(Image.open(src / "f-006.png").convert("RGB"))
    panels = {"5": ((0.0, 0.5, 0.05, 0.5), np.arange(200, 551, 50), np.arange(60, 111, 10)),
              "10": ((0.5, 1.0, 0.05, 0.5), np.arange(200, 601, 50), np.array([40, 60, 100, 120])),
              "15": ((0.0, 0.5, 0.5, 1.0), np.arange(200, 601, 50), np.arange(20, 121, 20)),
              "20": ((0.5, 1.0, 0.5, 1.0), np.arange(200, 601, 50), np.arange(40, 141, 20))}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (e, (box, xt, yt)) in zip(axes.ravel(), panels.items()):
        sub = dg.crop(img, *box)
        px, py = axis_maps(sub, xt, yt)
        d = figs["fig4"][e]
        k = d["atm"] + np.asarray(d["offsets"])
        overlay(ax, sub, px, py, k, {s: d[s] for s in dg.SKEW_COLOURS}, f"Figure 4, {e}y: digitised points")
    fig.tight_layout()
    fig.savefig(dst / "overlay_fig4.png", dpi=110)

    # Figure 8
    img = np.asarray(Image.open(src / "f-014.png").convert("RGB"))
    px, py = axis_maps(img, np.arange(200, 601, 100), np.arange(55, 71, 2.5))
    d = figs["fig8"]
    fig, ax = plt.subplots(figsize=(9, 6))
    overlay(ax, img, px, py, 387.0 + np.asarray(d["offsets"]), {s: d[s] for s in dg.ETA_COLOURS},
            "Figure 8: digitised points")
    fig.tight_layout()
    fig.savefig(dst / "overlay_fig8.png", dpi=110)

    # Figures 3, 6, 7, 9 (maturity on the x axis, frame = 0..40y)
    for name, suffix, ticks in (("fig3", "004", np.arange(0, 301, 50)), ("fig6", "010", np.arange(0, 176, 25)),
                                ("fig7", "012", np.arange(0, 121, 20)), ("fig9", "016", np.arange(0, 201, 50))):
        img = np.asarray(Image.open(src / f"f-{suffix}.png").convert("RGB"))
        px, py = axis_maps(img, None, ticks, x_limits=(0.0, 40.0))
        d = figs[name]
        t = np.asarray(d["T"])
        fig, ax = plt.subplots(figsize=(9, 6))
        sel = slice(7, None, 8)
        overlay(ax, img, px, py, t[sel], {s: np.asarray(v, dtype=object)[sel].tolist()
                                          for s, v in d.items() if s != "T"}, f"{name}: digitised points")
        fig.tight_layout()
        fig.savefig(dst / f"overlay_{name}.png", dpi=110)
    print("written to", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
