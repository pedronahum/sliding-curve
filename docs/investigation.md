# Investigation of the open issues

Status of every issue in the replication of Piterbarg & Van Der Helm, *Term
Premia and Bond Convexity* (6 Nov 2026). Each section gives the status, the
evidence, and a table of the variants tried with their effect on what
already matched (Figures 2, 3, 5, 6, 9-13 and `sigma = 1.157%`).

Status vocabulary: **fixed** (with a test that would have caught it),
**built** (with tests), **explained** (the difference is understood and
quantified; nothing in the code was wrong), **paper inconsistency** (written
up in `questions_for_authors.md`), **unresolved** (with the evidence that
would settle it).

Conventions. All numbers are basis points unless stated. "s.e." is a
Monte-Carlo standard error from this code; differences under two s.e. are
not read as signal. Seeds are fixed in every script. Each experiment is a
script under `experiments/` that writes `experiments/results/<name>.json`;
the tables below are copied from their output. Figures digitised from the
paper are accurate to about 0.3bp (checked below).

| ID | Issue | Status |
|---|---|---|
| A9 | Definition of the yield option | **pinned down** except one bounded choice; question to authors |

---

## Digitisation check (all A items)

`experiments/digitisation_overlay.py` maps every digitised value back to
pixels with the digitiser's own axis calibration and draws it on the
extracted figure image. Every point of the four panels of Figure 4, of
Figure 8 and of Figures 3, 6, 7 and 9 sits on the paper's own marker or
line. Digitisation error is ruled out as a cause of any of the A items.

![Figure 4 overlay](investigation/overlay_fig4.png)

Overlays of the other figures: [Figure 8](investigation/overlay_fig8.png),
[Figure 3](investigation/overlay_fig3.png), [Figure 6](investigation/overlay_fig6.png),
[Figure 7](investigation/overlay_fig7.png), [Figure 9](investigation/overlay_fig9.png).

---

## A9. What is the "yield option"?

**Status: pinned down, except whether strikes are offsets from the model
mean or from the curve forward; the two readings are bounded and both
are now available (`strike_reference=`).** The default stays "mean".

The paper never defines the option it calls a yield option (Figures 4, 8,
11; Section 9.4). Four choices are needed: the underlying, the measure
(equivalently the payment date), the forward around which implied vols are
inverted, and where strikes are measured from. Each is pinned by a different
published number.

**Underlying: the continuously compounded forward yield `Y(T, T, T+5)`.**
`experiments/a9_instrument_vols.py`, exact Gaussian prices (Jamshidian):

| 20y x 5y underlying | normal vol at sigma = 1% (baseline curve) | sigma for 82bp (market curve) |
|---|---|---|
| continuously compounded yield | 70.86 | **1.1572%** |
| simple rate L(T, T+5) (caplet) | 85.92 | 0.9267% |
| par swap, annual fixed | 73.80 | 1.1031% |
| par swap, semi-annual | 72.39 | 1.1279% |
| par swap, quarterly | 71.69 | 1.1405% |

The paper's caption to Figure 13 gives `sigma = 1.1572%`: five significant
figures, matched only by the yield. At `b = 0` its variance is the same in
every measure, so this does not yet fix the measure.

**Measure: the (T + delta)-forward measure** (payment at the end of the
window, as for the calibration caplets). `experiments/a9_measure_scan.py`
prices Figure 11 at the paper's own stated parameters (`sigma = 1.1999%`,
`b = 0.367`, no fitting) in the tau-forward measure for tau from T to T+5
and compares with the fitted vols the paper plots:

| tau | Figure 11: model minus paper's fitted vols at -200, -50, 0, +50, +200 |
|---|---|
| T | +11.6 +12.9 +13.2 +13.4 +14.1 |
| T + 1 | +8.8 +10.0 +10.3 +10.5 +11.0 |
| T + 2.5 | +5.1 +6.2 +6.4 +6.5 +6.9 |
| T + 4 | +1.8 +2.8 +2.9 +3.0 +3.2 |
| **T + 5** | **-0.1 +0.7 +0.8 +0.9 +1.0** (s.e. 0.6) |

The physical measure is excluded outright: there `x` has no mean reversion,
and the 20y at-the-money vol is 93bp at `sigma = 1%`, against 71.5bp in
Figure 8 (`experiments/a4_fig8_measures.py`).

**Inversion forward: the model's own mean of the yield in that measure.**
Inverting around the curve forward `Y(0, T, T+5)` instead gives 6.7-9.1bp
errors against Figure 8 at every eta, and prices below intrinsic at
low strikes (`experiments/a4_fig8_measures.py`, `a1_conventions.py`).

**Strikes: ambiguous between the model mean and the curve forward.** The
paper draws its at-the-money line at the curve forward in all three figures
(Figure 4: 370, 390, 395.5, 387.5 against forwards 369.1, 389.6, 395.3,
387.4; Figure 8: 387 against 387.4; Figure 11: 445 against 446.2). The two
figures that can discriminate disagree:

| | strikes from the model mean | strikes from the curve forward |
|---|---|---|
| Figure 11 at the paper's parameters, model minus paper's fit | -0.4 to +0.7 | +1.8 to +2.5 |
| Figure 8, mean abs error, eta = 0 / 0.5 / 1 / 1.5 | 0.39 / 0.58 / 1.32 / 1.23 | 0.39 / 0.52 / 1.50 / 0.63 |
| Figure 8, lowest plotted point (paper: at the money) | +50bp | at the money |

Figure 11 is the stronger evidence (a 2bp level difference, three s.e.;
Figure 8's preference rests on the location of a shallow minimum and on
differences of a few tenths of a basis point), and it is the figure that
feeds Figures 12 and 13, so the default remains strikes from the mean.
Effect of the choice on the skew fit (A6): `sigma` 1.190% against 1.165%,
`b` 0.354 against 0.372, `gamma(30y)` 135.5 against 129.5bp.

**Code changes.** `pricing.yield_option_prices` documents the reading and
gains `strike_reference="mean" | "forward"`; `yield_option_moneyness`
gives the strike to invert at; `calibration.calibrate_yield_smile`
evaluates its target and vega at that moneyness. Defaults unchanged, so no
existing result moves.

**Tests** (`tests/test_yield_option.py`): the Gaussian smile is flat at the
closed form for either reference; only the continuous yield gives 1.1572%;
the paper's stated skew parameters reproduce its Figure 11 fit with
mean-referenced strikes and miss with forward-referenced ones; 1.12% misses
by 3-5bp.
