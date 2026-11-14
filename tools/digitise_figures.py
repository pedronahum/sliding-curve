"""Read curves off the paper's raster figures.

The paper tabulates neither its synthetic curves nor the plotted results, so
this script recovers them from the figure images. Extract the images first:

    pdfimages -png paper.pdf figs/f
    python tools/digitise_figures.py figs slidingcurve/data

Method: locate the plot frame (dark spines) and the horizontal grid lines,
map pixel rows to values using the known tick labels, then for each pixel
column take the centroid of the pixels matching a line's colour. Accuracy is
about a quarter of a pixel, i.e. roughly 0.1-0.3bp on these axes, except where
one line hides another.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

COLOURS = {"black": (0, 0, 0), "royalblue": (65, 105, 225), "seagreen": (46, 139, 87),
           "goldenrod": (218, 165, 32), "tomato": (255, 99, 71)}

# file suffix -> (y tick values on the left axis, {series name: colour})
FIGURES = {
    # Figures 2 and 5: theta and gamma are on a right axis with the left
    # axis' scale, offset so that gamma(0) = 0 (subtracted below)
    "002": ("fig2", np.arange(300, 476, 25), {"Y": "black", "Y_eh": "royalblue",
                                               "theta": "tomato", "gamma": "seagreen"}),
    "008": ("fig5", np.arange(300, 551, 50), {"Y": "black", "Y_eh": "royalblue",
                                               "theta": "tomato", "gamma": "seagreen"}),
    "004": ("fig3", np.arange(0, 301, 50), {"b=0.00": "black", "b=0.25": "royalblue",
            "b=0.50": "seagreen", "b=0.75": "goldenrod", "b=1.00": "tomato"}),
    "010": ("fig6", np.arange(0, 176, 25), {"b=0.00": "black", "b=0.25": "royalblue",
            "b=0.50": "seagreen", "b=0.75": "goldenrod", "b=1.00": "tomato"}),
    "012": ("fig7", np.arange(0, 121, 20), {"eta=0.0": "black", "eta=0.5": "royalblue",
            "eta=1.0": "seagreen", "eta=1.5": "tomato"}),
    "016": ("fig9", np.arange(0, 201, 50), {"eta=0.0": "black", "eta=0.5": "royalblue",
            "eta=1.0": "seagreen", "eta=1.5": "tomato"}),
}
SKEW_COLOURS = {"b=0.00": "black", "b=0.25": "royalblue", "b=0.50": "seagreen",
                "b=0.75": "goldenrod", "b=1.00": "tomato"}
ETA_COLOURS = {"eta=0.0": "black", "eta=0.5": "royalblue", "eta=1.0": "seagreen", "eta=1.5": "tomato"}
X_RANGE = (0.0, 40.0)

# Section 9.4 decompositions: maturities 0-30y; theta and gamma are on a right
# axis with the same scale as the left one, offset so that gamma(0) = 0.
MARKET_FIGURES = {"018": "fig10", "022": "fig12"}
MARKET_TICKS = np.arange(350, 651, 50)
MARKET_SERIES = {"Y": "black", "Y_eh": "royalblue", "theta": "tomato", "gamma": "seagreen"}
# Figure 11: the five smile quotes (black circles) and the paper's fit (red squares)
SMILE_STRIKES = np.array([245.0, 395.0, 445.0, 495.0, 645.0])


def frame(img):
    dark = img.sum(axis=2) < 330
    cols = np.where(dark.mean(axis=0) > 0.5)[0]
    rows = np.where(dark.mean(axis=1) > 0.5)[0]
    return cols.min(), cols.max(), rows.min(), rows.max()       # left, right, top, bottom


def grid_rows(img, left, right, top, bottom):
    """Rows of the horizontal grid lines: light grey across most of the width."""
    inner = img[top + 3:bottom - 2, left + 3:right - 2].astype(float)
    grey = (np.abs(inner - inner.mean(axis=2, keepdims=True)).max(axis=2) < 6) & \
           (inner.mean(axis=2) < 245) & (inner.mean(axis=2) > 180)
    score = grey.mean(axis=1)
    rows = np.where(score > 0.5)[0]
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    return np.array([g.mean() for g in groups]) + top + 3


def trace(img, colour, left, right, top, bottom, legend):
    target = np.array(COLOURS[colour], dtype=float)
    dist = np.sqrt(((img.astype(float) - target) ** 2).sum(axis=2))
    w = np.clip(1.0 - dist / 90.0, 0.0, None)
    w[:top + 3] = 0; w[bottom - 2:] = 0
    lx, ly = legend
    w[:top + ly, :left + lx] = 0                                 # legend box
    xs, ys = [], []
    for col in range(left + 3, right - 1):
        c = w[:, col]
        if c.max() < 0.75:
            continue
        peak = c.argmax()
        lo, hi = max(peak - 5, 0), peak + 6
        seg = c[lo:hi]
        xs.append(col); ys.append((seg * np.arange(lo, hi)).sum() / seg.sum())
    return np.array(xs), np.array(ys)


def grid_cols(img, left, right, top, bottom):
    return grid_rows(np.transpose(img, (1, 0, 2)), top, bottom, left, right)


def market_figure(img):
    left, right, top, bottom = frame(img)
    rows = grid_rows(img, left, right, top, bottom)
    assert len(rows) == len(MARKET_TICKS) - 2, len(rows)         # 350 and 650 are the frame
    slope, intercept = np.polyfit(rows[::-1], MARKET_TICKS[1:-1], 1)
    grid_t = np.arange(0.25, 30.01, 0.25)
    res = {"T": grid_t.tolist()}
    for label, colour in MARKET_SERIES.items():
        xs, ys = trace(img, colour, left, right, top, bottom, legend=(150, 160))
        t = (xs - left) / (right - left) * 30.0
        vals = np.interp(grid_t, t, slope * ys + intercept, left=np.nan, right=np.nan)
        vals[[np.min(np.abs(t - g)) > 0.2 for g in grid_t]] = np.nan
        if np.isnan(vals[-1]) and not np.isnan(vals[-3:-1]).any():
            vals[-1] = 2 * vals[-2] - vals[-3]
        res[label] = vals
    offset = res["gamma"][0]                                     # gamma(0.25y) is ~0.01bp
    for label in ("theta", "gamma"):
        res[label] = res[label] - offset
    return {k: v if k == "T" else [None if np.isnan(x) else round(float(x), 2) for x in v]
            for k, v in res.items()}


def smile_figure(img):
    left, right, top, bottom = frame(img)
    rows, cols = grid_rows(img, left, right, top, bottom), grid_cols(img, left, right, top, bottom)
    ys, yi = np.polyfit(rows[::-1], np.arange(75, 96, 5), 1)
    xs, xi = np.polyfit(cols, np.arange(250, 651, 50), 1)
    res = {"strikes": SMILE_STRIKES.tolist()}
    for label, colour in (("market", "black"), ("paper_fit", "tomato")):
        dist = np.sqrt(((img.astype(float) - np.array(COLOURS[colour], dtype=float)) ** 2).sum(axis=2))
        w = np.clip(1.0 - dist / 90.0, 0.0, None)
        w[:top + 3] = 0; w[bottom - 2:] = 0; w[:top + 110, :left + 300] = 0
        vals = []
        for k in SMILE_STRIKES:
            col = int(round((k - xi) / xs))
            block = w[:, col - 1:col + 2].sum(axis=1)
            vals.append(round(float(ys * (block * np.arange(len(block))).sum() / block.sum() + yi), 2)
                        if block.sum() > 3 else None)
        res[label] = vals
    # where the fit's marker sits on top of the market marker the two coincide
    # to within the reading error; the hidden one cannot be read on its own
    res["market"] = [f if m is None or abs(m - f) < 0.4 else m
                     for m, f in zip(res["market"], res["paper_fit"])]
    return res


def read_panel(img, x_ticks, y_ticks, colours, xs, legend, x_limits=None):
    """Read each coloured line of one panel at the abscissae `xs`.

    x_ticks / y_ticks are the values of the interior grid lines; `legend` is
    (x0, x1, y0, y1) in fractions of the plot box to ignore. If x_limits is
    given the frame is used for the x axis instead of the grid lines. Columns
    crossed by a vertical line of the same colour (the dashed means of Figure
    13) are skipped and reported."""
    left, right, top, bottom = frame(img)
    rows = grid_rows(img, left, right, top, bottom)
    assert len(rows) == len(y_ticks), (len(rows), len(y_ticks))
    ys, yi = np.polyfit(rows[::-1], y_ticks, 1)
    if x_limits is None:
        cols = grid_cols(img, left, right, top, bottom)
        assert len(cols) == len(x_ticks), (len(cols), len(x_ticks))
        xs_, xi = np.polyfit(cols, x_ticks, 1)
    else:
        xs_ = (x_limits[1] - x_limits[0]) / (right - left)
        xi = x_limits[0] - xs_ * left
    lx0, lx1, ly0, ly1 = legend
    res, verticals = {}, {}
    for label, colour in colours.items():
        dist = np.sqrt(((img.astype(float) - np.array(COLOURS[colour], dtype=float)) ** 2).sum(axis=2))
        w = np.clip(1.0 - dist / 90.0, 0.0, None)
        w[:top + 3] = 0; w[bottom - 2:] = 0; w[:, :left + 3] = 0; w[:, right - 2:] = 0
        w[top + int(ly0 * (bottom - top)):top + int(ly1 * (bottom - top)),
          left + int(lx0 * (right - left)):left + int(lx1 * (right - left))] = 0
        tall = np.where((w > 0.5).sum(axis=0) > 40)[0]
        verticals[label] = round(float(xs_ * tall.mean() + xi), 1) if len(tall) else None
        pts_x, pts_y = [], []
        for col in range(left + 3, right - 2):
            c = w[:, col]
            if c.max() < 0.75 or (len(tall) and np.min(np.abs(tall - col)) < 3):
                continue
            peak = c.argmax()
            lo, hi = max(peak - 5, 0), peak + 6
            pts_x.append(xs_ * col + xi)
            pts_y.append(ys * (c[lo:hi] * np.arange(lo, hi)).sum() / c[lo:hi].sum() + yi)
        pts_x, pts_y = np.array(pts_x), np.array(pts_y)
        step = abs(xs_) * 3
        res[label] = [None if (len(pts_x) == 0 or np.min(np.abs(pts_x - x)) > step)
                      else float(np.interp(x, pts_x, pts_y)) for x in xs]
    return res, verticals


def crop(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def extra_figures(src):
    out = {}
    rnd = lambda d, n=2: {k: [None if v is None else round(v, n) for v in vals] for k, vals in d.items()}
    # Figure 8: smiles by eta, 20y x 5y; strikes are at-the-money +- 50k
    img = np.asarray(Image.open(src / "f-014.png").convert("RGB"))
    strikes = 387.0 + 50.0 * np.arange(-5, 6)
    vals, _ = read_panel(img, np.arange(200, 601, 100), np.arange(55, 71, 2.5), ETA_COLOURS,
                         strikes, legend=(0.0, 0.3, 0.8, 1.0))
    out["fig8"] = {"offsets": (strikes - 387.0).tolist(), **rnd(vals)}
    # Figure 4: smiles by skew at four expiries; strikes are at-the-money +- 40k
    img = np.asarray(Image.open(src / "f-006.png").convert("RGB"))
    panels = {"5": ((0.0, 0.5, 0.05, 0.5), 370.0, np.arange(200, 551, 50), np.arange(60, 111, 10)),
              "10": ((0.5, 1.0, 0.05, 0.5), 390.0, np.arange(200, 601, 50),
                     np.array([40, 60, 100, 120])),      # the 80 grid line lies under the b = 0 curve
              "15": ((0.0, 0.5, 0.5, 1.0), 395.5, np.arange(200, 601, 50), np.arange(20, 121, 20)),
              "20": ((0.5, 1.0, 0.5, 1.0), 387.5, np.arange(200, 601, 50), np.arange(40, 141, 20))}
    out["fig4"] = {}
    for expiry, (box, atm, xt, yt) in panels.items():
        strikes = atm + 40.0 * np.arange(-5, 6)
        vals, _ = read_panel(crop(img, *box), xt, yt, SKEW_COLOURS, strikes, legend=(0.0, 0.42, 0.0, 0.2))
        out["fig4"][expiry] = {"atm": atm, "offsets": (strikes - atm).tolist(), **rnd(vals)}
    # Figure 13: short-rate densities under P, without and with skew
    img = np.asarray(Image.open(src / "f-024.png").convert("RGB"))
    grid = np.arange(-1400.0, 1901.0, 50.0)
    dens = {"T=10": "royalblue", "T=20": "seagreen", "T=30": "tomato"}
    out["fig13"] = {"r": grid.tolist()}
    for name, box in (("atm", (0.0, 0.51, 0.08, 1.0)), ("skew", (0.51, 1.0, 0.08, 1.0))):
        vals, means = read_panel(crop(img, *box), None, np.arange(0, 0.00141, 0.0002), dens, grid,
                                 legend=(0.8, 1.0, 0.0, 0.22), x_limits=(-1500.0, 2000.0))
        out["fig13"][name] = {"density": rnd(vals, 7), "mean": means}
    return out


def main(src, dst):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for suffix, (name, ticks, series) in FIGURES.items():
        img = np.asarray(Image.open(src / f"f-{suffix}.png").convert("RGB"))
        left, right, top, bottom = frame(img)
        rows = grid_rows(img, left, right, top, bottom)
        assert len(rows) == len(ticks), (name, len(rows), len(ticks))
        slope, intercept = np.polyfit(rows[::-1], ticks, 1)      # rows decrease as values rise
        assert np.max(np.abs(slope * rows[::-1] + intercept - ticks)) < 0.3, name
        grid_t = np.arange(0.25, 40.01, 0.25)
        out[name] = {"T": grid_t.tolist()}
        for label, colour in series.items():
            xs, ys = trace(img, colour, left, right, top, bottom, legend=(330, 160))
            if len(xs) == 0:                                     # entirely hidden by other lines
                out[name][label] = [None] * len(grid_t)
                continue
            t = X_RANGE[0] + (xs - left) / (right - left) * (X_RANGE[1] - X_RANGE[0])
            v = slope * ys + intercept
            vals = np.interp(grid_t, t, v, left=np.nan, right=np.nan)
            # mark stretches with no pixels of this colour (hidden by another line)
            gap = np.array([np.min(np.abs(t - g)) > 0.2 for g in grid_t])
            vals[gap] = np.nan
            if np.isnan(vals[-1]) and not np.isnan(vals[-3:-1]).any():
                vals[-1] = 2 * vals[-2] - vals[-3]      # the line ends under the right spine
            out[name][label] = [None if np.isnan(x) else round(float(x), 2) for x in vals]
        if "gamma" in series:                                    # right-axis offset: gamma(0.25y) ~ 0.01bp
            off = out[name]["gamma"][0]
            for label in ("theta", "gamma"):
                out[name][label] = [None if x is None else round(x - off, 2) for x in out[name][label]]
    for suffix, name in MARKET_FIGURES.items():
        out[name] = market_figure(np.asarray(Image.open(src / f"f-{suffix}.png").convert("RGB")))
    out["fig11"] = smile_figure(np.asarray(Image.open(src / "f-020.png").convert("RGB")))
    out.update(extra_figures(src))
    (dst / "paper_figures_digitised.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    out = main(*sys.argv[1:3])
    print("fig11", out["fig11"])
    print("fig8", out["fig8"])
    for e, d in out["fig4"].items():
        print("fig4", e, {k: v for k, v in d.items() if k != "offsets"})
    for k in ("atm", "skew"):
        d = out["fig13"][k]
        print("fig13", k, "means", d["mean"], "T=30 density", d["density"]["T=30"][::8])
    for name, d in out.items():
        if name in ("fig11", "fig8", "fig4", "fig13"):
            continue
        idx = [3, 19, 39, 79, 119, 159] if len(d["T"]) > 120 else [3, 19, 39, 79, 99, 119]
        print(name, "T =", [d["T"][i] for i in idx])
        for k, v in d.items():
            if k != "T":
                print(f"   {k:8s}", [v[i] for i in idx], " missing:", sum(x is None for x in v))
