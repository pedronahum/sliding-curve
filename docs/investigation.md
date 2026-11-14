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
| A4 | Figure 8 minimum and eta = 0 level | **fixed** (minimum: comparison axis); **paper inconsistency** (level) |
| A5 | Section 8.2 uplift | **explained** |
| A8 | Curve beyond the plotted range | **unresolved by construction**, bounded; question to authors |
| A10 | Two values of sigma for the skew fit | **paper inconsistency**, confirmed |

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

---

## A10. Two values of sigma for the skew fit

**Status: paper inconsistency, confirmed.** The text of Section 9.4 says
the skew fit gives `sigma = 1.12%`; the caption of Figure 13 says 1.1999%.
At `b = 0.367` (`experiments/a6_skew_fit.py`, `a10_text_sigma.py`):

| sigma | Figure 11: model minus paper's fitted vols | gamma(30y), paper 140.3 |
|---|---|---|
| 1.1999% | -0.4 to +0.7 (mean strikes) | 140.5 +- 0.2 |
| 1.12% | -4.5 to -5.3 (mean strikes); -3.0 to -3.9 (forward strikes) | 115.6 +- 0.2 |

1.12% does not reproduce Figure 11 under either strike reading, nor Figure
12. It is not the skew-fit sigma of any of the instrument readings in A9
either (the annual par swap would need 1.10% at `b = 0`). The test
`test_papers_skew_parameters_reproduce_its_figure_11_fit_with_mean_referenced_strikes`
checks both values.

---

## A4. Figure 8: position of the minimum and the eta = 0 level

**Status: minimum fixed (it was a strike-axis artefact in the comparison,
not in the model); the eta = 0 level is a paper inconsistency, most
plausibly Monte-Carlo noise in the paper's own run.**

The fix: `examples/stochvol_sweep.py` now plots both smiles against
absolute strike (this code's at mean + offset, the paper's at its
at-the-money line, 387bp, + offset), and
`test_figure_8_smile_minimum_on_absolute_strikes` checks that the eta = 1.5
smile bottoms out within 20bp of the paper's minimum (it is within 5bp; on
the old axis it would have read 27bp off).

*Minimum.* The old comparison plotted this code's strikes as offsets from
its own mean (364bp at 20y) against the paper's offsets from the curve
forward (387bp). On the paper's absolute strike axis the lowest point of
this code's smile is where the paper's is (A9 table above); the parabola
through the three lowest points puts the paper's minimum at 400-405bp and
this code's at 404-407bp. In the expiry-forward measure the minimum would
be at 365-370bp, so the comparison also confirms the measure.

*Level.* At `b = 0`, `eta = 0`, `sigma = 1%` the option is Gaussian and its
vol is exactly 70.86bp. Figure 8 plots 71.4 (text: 71.5) and its eta = 0
line is not flat: it tilts from 71.2 to 71.6bp across strikes. No pricing
convention makes a Gaussian smile tilt (A1 below), but a Monte-Carlo
estimate does, since all strikes share the same paths.
`experiments/a4_fig8_measures.py` measures the spread of the Gaussian
at-the-money vol estimate for plain (non-antithetic) paths:

| paths | spread of the ATM vol estimate | P(abs error >= 0.55bp) |
|---|---|---|
| 4,096 | 0.79bp | 0.41 |
| 16,384 | 0.40bp | 0.20 |
| 65,536 (this code, antithetic) | 0.33bp quoted s.e. | |

So the paper's level is what a run with 4k-16k paths produces a fifth to
two fifths of the time. Recalibrating Figure 9 to the paper's 71.5bp
instead of the exact level was tried (A3): it breaks the match at eta = 0.5
(136.1 against 133.6bp), so the paper's Figure 9 is not consistent with
71.5bp as an exact target either. Only the path count and seed of the
paper's run can settle this.

---

## A5. Section 8.2: uplift for eta = 0.6 over ten years

**Status: explained.** Section 8.2 computes the uplift from the Jensen gap
(65) of a driftless rate run on a stochastic clock; this code reports the
full model recalibrated to an at-the-money option. They are different
quantities and both are right. `experiments/a5_uplift.py` (variance weights
`exp(-2 kappa (T - t))`, i.e. the weight with which variance at t reaches
the rate at expiry, `kappa_z = 0.1`, `T = 10`):

| kappa | relative variance | 1 + rel. var / 4 | **exact E(V) / E(sqrt V)^2** | paper | full model, yield option | full model, caplet |
|---|---|---|---|---|---|---|
| 0 | 0.605 | 15.1% | **14.52 +- 0.03%** | ~15% | 13.49% | 14.16% |
| 0.03 | 0.678 | 17.0% | **16.27 +- 0.03%** | ~16% | 15.41% | 15.79% |
| 0.05 | 0.728 | 18.2% | **17.51 +- 0.04%** | 17.5% | 16.73% | 16.99% |

The paper's numbers are the exact Jensen ratio (not the second-order
approximation, which overstates by 0.6-0.7 points); its ς² ≈ 0.6 is our
0.605. The full-model columns above are single runs; repeated over four
seeds at 131,072 paths the kappa = 0 yield-option uplift is 13.85 +- 0.07%
(single-run spread 0.15), so the 13.49% of a single 65,536-path run was a
low draw.

Where the remaining 0.6-0.7 points go (`experiments/a5_drift_attribution.py`,
same variance paths and rate noise throughout):

| kappa | exact Jensen | model rate made driftless | + forward-measure drift (y term, -sigma_r^2 B) | same, sigma solved exactly | full model |
|---|---|---|---|---|---|
| 0 | 14.52% | 14.44% | 14.15% | 14.09% | 13.85 +- 0.07% |
| 0.03 | 16.27% | 16.17% | 15.99% | 15.94% | |
| 0.05 | 17.51% | 17.41% | 17.25% | 17.21% | |

The driftless model rate reproduces the Jensen ratio (to 0.1 points of
time discretisation). The drift that the benchmark yield has in the forward
measure is itself driven by the realised variance and takes another 0.3-0.4
points; the rest (0.2 points at kappa = 0) is within 1.6 single-run
standard deviations. Nothing is wrong on either side: Section 8.2 states the
idealised effect, the model includes the drift. The README previously set
one against the other.

---

## A8. The curve beyond the plotted range

**Status: unresolved by construction, bounded.** Y_eh and theta at the last
plotted maturity depend on forwards up to five years beyond it, which no
figure shows. `experiments/a8_extrapolation.py` tries four extensions of the
digitised zero curve (all reproduce the digitised yields inside the plotted
range to 0.4-0.6bp):

| market curve, T = 30y | quadratic (library) | linear | flat zero | flat forward | paper |
|---|---|---|---|---|---|
| Y_eh | 377.9 | 373.6 | 370.4 | 373.8 | 376.4 |
| theta | 173.1 | 177.4 | 180.6 | 177.2 | 174.7 |
| gamma | 107.8 | 107.8 | 107.8 | 107.8 | 108.0 |

| baseline curve, T = 40y | quadratic | linear | flat zero | flat forward | paper |
|---|---|---|---|---|---|
| Y_eh | 304.9 | 303.2 | 300.0 | 303.2 | 304.8 |

Convexity does not depend on the extension at `b = 0`. Y_eh and theta move
together by up to 7bp in opposite directions (their sum is fixed by the
curve). The paper's 30y values lie inside the band, between the quadratic
and the linear extension, and its 40y Y_eh on the baseline curve matches
the quadratic one to 0.1bp. The 1.5bp residual at 30y is therefore within
what the unpublished part of the curve leaves open; only the authors' curve
beyond 30 (and 40) years can close it. Inside the plotted range, at 20y and
25y, all four extensions agree with each other and with the paper to 0.3bp.
