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
| A1 | Figure 4 | **paper inconsistency** (contradicts the text, Figure 8 and the model's Gaussian limit); question to authors |
| A2 | Convexity at high skew | **partly fixed** (bootstrap grid); rest bounded by the cap definition; question to authors |
| A3 | Figure 9 at eta = 1.5 | **unresolved**, bounded; question to authors |
| A4 | Figure 8 minimum and eta = 0 level | **fixed** (minimum: comparison axis); **paper inconsistency** (level) |
| A5 | Section 8.2 uplift | **explained** |
| A6 | Skew fit | **explained** to within Monte-Carlo noise; question to authors |
| A7 | Three-parameter fit | **explained**: eta not identified at Monte-Carlo precision; question to authors |
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

---

## A3. Figure 9 at eta = 1.5

**Status: unresolved, bounded.** This code gives 202.7 +- 1.3bp at 40y where
the paper plots 212.8bp; eta = 0.5 and 1.0 match (133.7 vs 133.6, 165.5 vs
166.7).

*The README's hypothesis is refuted.* It said the paper's convexity ratios
equal the squared ratios of its at-the-money vols, as if sigma were
rescaled linearly. They do not: at eta = 0.5 the paper's ratio is 1.121
against a squared vol ratio of 1.166 (1.169 with 71.5bp). This code's exact
recalibration gives 1.124 there, matching the paper.

*Variants, one at a time* (`experiments/a3_stochvol_variants.py`; sigma
recalibrated at each eta to the eta = 0 ATM vol; gamma(40y), paper 133.6 /
166.7 / 212.8):

| variant | eta = 0.5 | eta = 1.0 | eta = 1.5 |
|---|---|---|---|
| library (QE, cap 10x, 24 / 12 steps a year) | 133.7 | 165.5 | 202.7 |
| pricing seed 1 / 2 / 3 | 133.5 / 133.3 / 134.3 | 165.0 / 164.9 / 168.0 | 203.8 / 202.3 / 208.0 |
| recalibrate to 71.5bp (paper's text) | **136.1** | 168.4 | 206.2 |
| no cap; cap in pricing only; in decomposition only; 20x | 133.7 | 165.5 | 202.7 |
| cap 5x | 133.7 | 165.5 | 201.0 |
| 52 steps a year in both | 133.8 | 165.0 | 203.8 |
| full-truncation Euler for z | 133.7 | 166.0 | 205.0 |
| reflection Euler for z | 133.7 | **153.2** | **160.1** |

- The cap never binds at `b = 0` for eta <= 1.5 (`sqrt(z) > 10` is too rare
  to move any digit), so neither its level nor where it is applied matters.
- Reflection inflates E[z], so the at-the-money vol *rises* with eta (85.8bp
  at eta = 1.5): ruled out by Figure 8.
- Recalibrating to 71.5bp breaks eta = 0.5.
- No variant reaches 212.8 without breaking eta = 0.5 or 1.0.

*Variance scheme against exact sampling.* `experiments/a3_exact_cir.py`
drives the same rate dynamics with exact CIR draws (noncentral chi-square)
and with the two schemes, four independent seeds of 131,072 paths each,
20y x 5y ATM vol at `sigma = 1%`, `b = 0`:

| eta | exact CIR | QE (library) | lognormal (B1) | paper (Figure 8) |
|---|---|---|---|---|
| 1.0 | 59.71 +- 0.07 | 59.60 +- 0.15 | 59.65 +- 0.16 | 60.87 |
| 1.5 | 53.32 +- 0.06 | 53.37 +- 0.18 | 53.46 +- 0.15 | 53.58 |

QE is unbiased at the 0.1bp level against exact sampling, so the scheme is
not the cause. The paper's Figure 8 sits +1.2bp and +0.3bp above these
precise values (and -0.7bp below this code's single run at eta = 0.5):
non-monotone in eta, as independent sampling errors would be.

*Noise.* The quoted s.e. (1.3bp) covers the decomposition only. The
recalibrated sigma inherits the pricing noise: over four pricing seeds
gamma(40y) is 202.7, 203.8, 202.3, 208.0 (mean 204.2 +- 1.3, spread 2.6bp).
The paper is 8.6bp above that mean: about three single-run standard
deviations at 65,536 paths, so not explained by this code's noise.
If the paper's runs used 4k-16k paths, as its eta = 0 level suggests (A4),
its own single-run spread would be 5-10bp and 212.8 would be within two of
them. That is the only explanation left standing and it cannot be checked
without the paper's path count and seeds. Section B4 shows that a control
variate does not help here (the twin does not see z); a conditional
(Rao-Blackwell) estimator would.

---

## A6. The skew fit on the market smile

**Status: explained to within Monte-Carlo noise; the fitting weights and
path count are questions for the authors.**

*The paper's parameters are consistent with this code.* At `sigma =
1.1999%`, `b = 0.367` this code reproduces the paper's plotted fitted vols
to -0.4..+0.7bp (s.e. 0.6) and Figure 12's gamma(30y) to 140.5 +- 0.2
against 140.3 (A9, A10). The engine is not the issue; the fit is.

*Weights.* The paper says the fit matches the at-the-money quote exactly.
Equal weights miss it by 1.9bp, so the at-the-money quote is pinned (weight
30). Fits (`experiments/a6_skew_fit.py`, 65,536 paths):

| strikes from | weights | sigma | b | fitted vols | gamma(30y) |
|---|---|---|---|---|---|
| mean | equal | 1.2181% | 0.346 | 74.7 81.7 83.9 86.1 92.5 | 144.2 |
| mean | ATM pinned | 1.1897% | 0.354 | 72.9 79.8 82.0 84.2 90.5 | 135.5 |
| forward | equal | 1.1918% | 0.365 | 74.7 81.7 83.9 86.1 92.5 | 137.4 |
| forward | ATM pinned | 1.1652% | 0.372 | 72.9 79.8 82.0 84.2 90.5 | 129.5 |
| paper | | 1.1999% | 0.367 | 73.4 79.8 82.0 84.2 90.7 | 140.3 |

With the at-the-money quote pinned, the fitted smile matches the paper's
fitted smile to 0.45bp under either strike reading, yet sigma, b and gamma
differ: the five quotes fix a valley in (sigma, b) along which gamma(30y)
moves by about 10bp per basis point of smile level.

*Noise.* The same fit over six pricing seeds (`experiments/a6_fit_noise.py`):

| paths | sigma | b | gamma(30y) |
|---|---|---|---|
| 32,768 | 1.1888 +- 0.0095% | 0.3530 +- 0.0060 | 135.1 +- 2.7 |
| 65,536 | 1.1890 +- 0.0055% | 0.3526 +- 0.0027 | 135.1 +- 1.7 |
| paper | 1.1999% | 0.367 | 140.3 |

(+- is the spread of a single fit.) The paper's point is 1-2 single-fit
standard deviations away at 32k paths in sigma and gamma, and 2-5 in b.
The difference corresponds to a 0.5bp shift of the fitted smile level, the
size of one pricing s.e. of either code. With the control variate of B4 the
fit noise falls (see B4), which leaves the paper's own path count and
weights as the open question.

---

## A7. The three-parameter fit

**Status: explained. eta is not identified by the five quotes at
Monte-Carlo precision, which is the paper's own conclusion in different
words.** Both the paper's point and this code's lie on the same valley.

`experiments/a7_three_param.py` profiles the fit: eta fixed on a grid,
(sigma, b) fitted by autodiff with the at-the-money quote pinned, 32,768
paths, strikes from the mean.

| eta | sigma | b | errors at -200, -50, 0, +50, +200 | rms | gamma(30y) |
|---|---|---|---|---|---|
| 0 | 1.194% | 0.353 | -4.26 -0.04 0.01 -0.60 -4.77 | 2.87 | 136.9 +- 0.3 |
| 0.4 | 1.271% | 0.366 | -3.37 0.07 0.01 -0.58 -4.07 | 2.38 | 171.7 +- 1.3 |
| 0.6 | 1.365% | 0.382 | -2.70 0.11 0.01 -0.53 -3.54 | 2.00 | 223.9 +- 2.7 |
| **0.779** | 1.468% | 0.395 | -2.26 0.15 0.01 -0.54 -3.21 | 1.77 | 290.5 +- 4.0 |
| 1.0 | 1.597% | 0.399 | -1.98 0.23 0.01 -0.53 -2.82 | 1.56 | 377.0 +- 6.3 |
| 1.2 | 1.698% | 0.391 | -1.77 0.26 0.00 -0.53 -2.68 | 1.46 | 434.3 +- 7.8 |
| **1.39** | 1.787% | 0.372 | -1.61 0.33 0.00 -0.52 -2.46 | 1.34 | 457.5 +- 8.6 |
| 1.6 | 1.859% | 0.349 | -1.56 0.39 0.00 -0.61 -2.35 | 1.30 | 464.1 +- 9.4 |

| point | errors | rms | gamma(30y) |
|---|---|---|---|
| paper (1.492%, 0.417, 0.779) | -2.61 0.56 0.60 0.22 -2.01 (s.e. up to 1.27) | 1.52 | 315.6 +- 4.4 |
| this code's earlier fit (1.80%, 0.36, 1.39) | -0.84 1.15 0.84 0.31 -1.65 (s.e. up to 1.44) | 1.05 | 463.6 +- 8.6 |
| full fit started at sigma 2.6%, eta 2 | -0.99 1.16 0.78 0.18 -1.29 at (1.95%, 0.32, 2.27) | | |

- The rms error falls monotonically along the valley; it has no interior
  minimum. Beyond eta ~ 0.8 the whole improvement (1.77 to 1.30bp) is
  smaller than the pricing s.e. of a single quote (about 1.3bp with
  stochastic volatility), so where a fit "converges" is set by the noise
  of its random numbers. The paper's point is within two s.e. of the
  quotes, as is this code's.
- The "spurious basin" is not separate: it is the same valley continued to
  larger eta, where the cap starts to do work.
- What the valley does to convexity is the point the paper makes: gamma(30y)
  runs from 137 to 464bp across fits that are equally good to Monte-Carlo
  precision; at the paper's own point it is 316bp ("not economically
  plausible").

The B1 lognormal variance scheme makes this fit differentiable in eta (and
faster), but it cannot make eta identified. What would: more expiries or
the curvature of the smile further out, or far more paths (the s.e. has to
fall below the 0.5bp rms differences along the valley).

---

## A1. Figure 4 (yield-option smiles by skew)

**Status: paper inconsistency.** No reading of the option, of the
volatility scaling or of the local volatility reproduces Figure 4 at 10-20
years; the figure contradicts the paper's own text and its own Figure 8;
and its `b = 0` smiles are not those of any Gaussian model. Digitisation is
ruled out (overlay above).

*The figure against the text.* Change of the at-the-money vol from `b = 0`
to `b = 1`:

| expiry | paper's text | paper's Figure 4 | this code, strikes from the mean | from the forward |
|---|---|---|---|---|
| 5y | "around 4bp below" | 0.0 | -4.5 | -3.3 |
| 10y | | +1.5 | | |
| 15y | | +1.2 | | |
| 20y | "6 to 8bp" below | **+5.9** | -9.0 | -5.1 |

This code agrees with the text (to within the strike-reference ambiguity of
A9); the figure has the opposite sign.

*The figure against Figure 8.* The 20y `b = 0` smile of Figure 4 is the same
option as the eta = 0 smile of Figure 8 (`sigma = 1%`, `b = 0`, `eta = 0`).
Figure 8 has it flat at 71.4bp; Figure 4 has it at 68.4bp at the money,
drooping to 55.1bp at +120bp, with no points at all at +160 and +200bp.

*The b = 0 smiles are not Gaussian under any convention.* At `b = 0` the
yield is Gaussian with the same variance in every measure, so any pricing
convention gives `price = c * Bachelier(m, K, s)` inverted with some forward
F. `experiments/a1_gaussian_fingerprint.py` fits `c` and `m - F` freely to
the paper's `b = 0` points, for calls, puts and out-of-the-money options:

| expiry | Gaussian vol | best fit: rms, convention, c, m - F |
|---|---|---|
| 5y | 86.31 | 2.1bp: any (the paper is flat at 86.0; fine) |
| 10y | 80.53 | 0.4bp: puts, c = 0.98, -2.3bp |
| 15y | 75.40 | 1.6bp: puts, c = 0.92, -15.7bp |
| 20y | 70.86 | 3.2bp: puts, c = 0.80, -40.7bp |

Even with two free parameters the 15-20y shapes (flat, then a cliff, then
missing points) are not reached, and the parameters needed are not those of
any measure or numeraire (a 20% price scaling). A Gaussian smile cannot
lose points at +160bp at 20y, half a standard deviation from the money,
unless prices fall below intrinsic, i.e. unless the inversion forward and
the option's mean disagree (or the price is wrong).

*Every reading of the option* (`experiments/a1_conventions.py`; b = 0 to 1,
sigma = 1%, strikes on the paper's grid, forward + 40k bp). Mean absolute
error against Figure 4, and points missing (model / paper):

| measure / inversion forward | 5y | 10y | 15y | 20y |
|---|---|---|---|---|
| Q^(T+delta) / model mean (library) | 1.1 [0/3] | 7.1 [0/2] | 11.0 [0/6] | 15.6 [0/11] |
| Q^(T+delta) / curve forward | 7.6 [4/3] | 9.6 [2/2] | 12.7 [2/6] | 18.3 [2/11] |
| Q^T (= spot Q, discounted) / model mean | 1.9 | 6.5 | 9.7 | 12.4 |
| Q^T / curve forward | 9.8 | 13.3 | 15.6 | 16.1 |
| P / model mean | 7.1 | 14.3 | 19.4 | 21.6 |
| P / curve forward | 12.9 [12/3] | 11.5 [6/2] | 7.9 [2/6] | 11.9 [0/11] |
| P, discounted pathwise / curve forward | 18.1 | 26.9 | 33.4 | 42.7 |

(The "missing" counts for the paper include points hidden where lines
cross; at 20y the genuinely absent markers are `b = 0` at +160, +200, `b = 0.5`
at -200, -160 and `b = 1` at -200 to -120.)

*Every scaling and local-volatility variant* (`experiments/a1_scaling_variants.py`,
library measure). The slope of the 20y `b = 1` smile between -80 and +200bp
is 110bp in the paper:

| variant | error 5 / 10 / 15 / 20y | 20y b = 1 slope | ATM change b = 1 vs 0, 5y / 20y |
|---|---|---|---|
| library (annual unweighted bootstrap), strikes from mean | 1.2 / 7.2 / 11.2 / 16.3 | 25.4 | -4.5 / -9.0 |
| strikes from forward | 0.8 / 7.0 / 11.0 / 15.6 | 24.5 | -3.3 / -5.1 |
| zeroth-order scaling (79) | 1.8 / 7.9 / 12.6 / 18.9 | 22.7 | -5.5 / -15.5 |
| bootstrap, monthly grid | 1.0 / 7.1 / 11.2 / 16.3 | 25.4 | -3.9 / -9.1 |
| bootstrap, weighted (80) | 1.2 / 7.2 / 11.3 / 16.6 | 25.1 | -4.5 / -9.9 |
| floor on the rate, not the vol | 1.2 / 7.2 / 11.2 / 16.3 | 25.4 | -4.5 / -9.0 |
| relative vol flat (literal (49)) | 3.8 / 7.9 / 11.7 / 16.7 | 24.9 | -5.4 / -9.9 |

Nothing moves the slope off 25bp: the model's smile at `b = 1` is what a
lognormal-like local volatility on a 20% relative vol gives, and the
paper's is four times steeper. The 5y panel is matched (1.1bp) by the
library reading.

*What Figure 4 looks like.* The pattern (a positive skew steepening with
expiry, missing points where the vol falls, a drooping `b = 0` line) is
what one gets when option prices are inverted around a forward that does
not match the distribution priced, a mismatch that grows with expiry. We
could not find a mismatch that reproduces it quantitatively and do not
claim one. Only the authors' Figure 4 code can settle it.

---

## B1. Calibration with eta free: finite differences

**Status: fixed.** `Model(variance_scheme="lognormal")` is a variance step
that is smooth in eta, so calibrations with eta free use autodiff; QE stays
the default.

*Why finite differences were not sound.* The quadratic-exponential scheme
switches branch at psi = 1.5, so at fixed random numbers a path's variance
jumps as eta moves. Its autodiff derivative ignores the jumps, and finite
differences straddle them: on the paper's three-parameter point
(`experiments/b1_variance_schemes.py`, d price / d eta for three strikes) the
QE finite-difference derivative changes with the step (-0.00148 at h = 1e-3,
-0.00197 at h = 1e-2, -0.00151 at 1e-4 for the low strike), and disagrees
with QE autodiff (-0.00189).

*The scheme.* The moment-matched lognormal step of Andersen and
Brotherton-Ratcliffe: `z' = m exp(-s^2/2 + s xi)`, `s^2 = log(1 + psi)`, with
the same conditional mean `m` and variance as QE (and as exact CIR), so
`E[z] = 1` holds without time-step bias. Alternatives considered: exact
sampling (Poisson-gamma mixture; not differentiable in eta through the
Poisson count), likelihood-ratio estimators (unbiased but with variance
growing with the number of steps, 480 here), smoothing QE's branch switch
(biased at the switch). The lognormal step is the simplest that is both
smooth and moment-exact.

*Validation.*

| | exact CIR | QE | lognormal |
|---|---|---|---|
| E[z(20y)], eta = 1.5 | 0.988 +- 0.018 | 0.992 +- 0.018 | 0.986 +- 0.018 |
| Var z(20y) (exact 11.04) | 10.59 | 10.59 | 10.46 |
| E[int z] / sd | 19.78 / 40.8 | 19.69 / 41.0 | 19.68 / 41.0 |
| P(z(20y) < 1e-3) | 0.454 | 0.512 | 0.160 |
| 20y ATM vol, eta = 1.0 (4 seeds x 131k) | 59.71 +- 0.07 | 59.60 +- 0.15 | 59.65 +- 0.16 |
| 20y ATM vol, eta = 1.5 | 53.32 +- 0.06 | 53.37 +- 0.18 | 53.46 +- 0.15 |
| smile at the paper's 3-parameter point, -200..+200 | 74.7 .. 93.1 | 75.2 .. 93.8 | 75.1 .. 93.9 (within 0.03 of QE) |

The lognormal step puts less mass near zero (as expected of a lognormal),
but option prices and convexity depend on the integrated variance, whose
moments it reproduces; prices agree with exact sampling within the s.e.

*Derivative.* Lognormal autodiff against central differences: -0.0019525 /
-0.0044982 / -0.0045482 against -0.0019524 / -0.0044964 / -0.0045481 at
h = 1e-4: four digits. `tests/test_variance_schemes.py` checks the
conditional moments, autodiff against finite differences, agreement with QE,
and recovery of known (sigma, b, eta) by a fit that uses autodiff.

*Effect* (`experiments/b1_fit_timing.py`, the paper's smile, 32,768 paths,
same start): QE with finite differences 83s, ending at (1.752%, 0.379,
1.274); lognormal with autodiff 40s, ending at (1.710%, 0.380, 1.178).
Both stop on the same valley (A7), with errors within 0.1bp of each other.

---

## A2. Convexity at high skew (Figures 3 and 6)

**Status: partly fixed (the bootstrap grid was not converged); the remaining
1.7-2.6% shortfall lies inside the sensitivity to the volatility cap,
which the paper does not define precisely enough. Question to authors.**

*Hypotheses, one at a time* (`experiments/a2_skew_variants.py`; gamma at
40y, model minus paper, s.e. in brackets; library = annual unweighted
bootstrap, 12 steps a year, 65,536 paths, seed 1). The last three columns
are results that already matched:

| variant | base b=1 | base b=0.75 | steep b=1 | steep b=0.75 | base b=0.25 | steep b=0.25 | Fig. 12 at 30y |
|---|---|---|---|---|---|---|---|
| library | -4.5 (2.6) | -2.8 (1.6) | -6.3 (1.4) | -4.1 (0.8) | +0.2 | -0.7 | +0.2 |
| zeroth-order scaling (79) | -137.9 | -115.7 | -66.3 | -45.7 | -26.6 | -9.3 | -24.6 |
| weighted bootstrap (80) | -31.5 | -30.1 | -19.1 | -13.7 | -9.6 | -2.5 | -5.7 |
| **monthly bootstrap grid** | -4.8 | -3.9 | **-3.1** | **-2.2** | -0.5 | -0.3 | -0.5 |
| 24 / 52 steps a year | -1.4 / -2.4 | -0.4 / -2.0 | -3.9 / -4.3 | -2.6 / -3.1 | +0.6 / +0.3 | -0.6 | +0.5 / +0.3 |
| curve extension linear / flat / flat forward | -4.6 / -4.8 / -4.6 | | -6.3 / -6.2 / -6.3 | | -0.4 / -1.0 / -0.3 | -0.7 | -0.6 / -1.0 / -0.5 |
| vol not floored (abs) | -4.5 | -2.7 | -6.3 | -4.1 | +0.2 | -0.7 | +0.2 |
| rate floored instead of vol | -4.5 | -2.0 | -6.3 | -3.8 | **+3.1** | +1.0 | +1.6 |
| cap on the local part only | identical to library at eta = 0 | | | | | | |
| no cap | **+5660** | **+1041** | +38.0 | +2.3 | +1.9 | -0.7 | +1.0 |

Zeroth-order and weighted scalings are ruled out (they break everything that
matched); so is flooring the rate. Without the cap, b = 1 explodes.

*The bootstrap grid was not converged.* (81) is the rectangle-rule
discretisation of the continuous condition (80); refining its expiry grid
(`experiments/a2_grid_convergence.py`, common random numbers, only the
scaling differs) shows a first-order error:

| change from the annual grid | semi-annual | quarterly | monthly | fortnightly | weekly |
|---|---|---|---|---|---|
| steep, b = 1 | +1.74 | +2.62 | +3.21 | +3.35 | +3.42 |
| steep, b = 0.75 | +1.01 | +1.52 | +1.86 | +1.95 | +1.99 |
| baseline, b = 1 | -0.18 | -0.26 | -0.32 | -0.10 | -0.10 |
| baseline, b = 0.25 | -0.41 | -0.60 | -0.72 | -0.67 | -0.69 |
| market, b = 0.367 (Fig. 12, 30y) | -0.33 | -0.48 | -0.63 | -0.69 | -0.70 |

This is a numerical error of the discretised scaling, independent of any
figure, so it is fixed: `volscale.bootstrap_grid` (monthly, within 0.2bp of
the limit) is now the default in the calibrations, the paper-replication
examples and tests. The bootstrap was vectorised to make this affordable:
the recursion telescopes, `Ybar_m K_m = R_m - R_{m-1}` with
`R_m = sum_{k<=m} Yhat(T_k, T_m) K_k`; identical to the loop to 1e-12
(`test_vectorised_bootstrap_equals_the_recursion_81`).
`test_default_bootstrap_grid_is_converged` would have caught the annual
grid (3.4bp from the weekly one; the default is within 0.5bp). Effect on
what matched: Figure 12 140.8 -> 140.1 (paper 140.3); baseline b = 0.25
144.6 -> 143.9 (144.1); Figure 13 mean short rate at 30y with skew 274.8 ->
282.4 +- 1.6 (280.1, 262,144 paths); sigma = 1.157% unchanged (b = 0).

*Time step: no effect.* With 262,144 paths and two seeds per step size
(`experiments/a2_step_convergence.py`, monthly grid), 12, 24 and 48 steps a
year agree within about 1bp. The same runs give the precise remaining gap:

| | paper | 12 / yr | 24 / yr | 48 / yr |
|---|---|---|---|---|
| baseline, b = 0.5 | 202.1 | 198.7 +- 0.3 | 198.5 +- 0.3 | 199.0 +- 0.4 |
| baseline, b = 0.75 | 267.6 | 262.0 +- 0.6 | 261.2 +- 0.6 | 261.8 +- 0.6 |
| baseline, b = 1 | 323.3 | 317.2 +- 0.9 | 314.9 +- 0.9 | 315.5 +- 0.9 |
| steep, b = 1 | 186.3 | 183.1 +- 0.5 | 182.0 +- 0.5 | 182.2 +- 0.5 |

(The 65,536-path runs with seed 1 used above happened to be high draws.)

*The cap.* Remark 8 caps "the realized volatility in the simulations ... at
10x the initial volatility" to deal with moment explosion under stochastic
volatility. At high skew it binds even with eta = 0: in P the benchmark's
drift contains the convexity term, quadratic in the rate, and with b = 1 the
local vol is proportional to the rate, so without a cap the second moment
blows up. Sensitivity of gamma(40y) to the cap level
(`experiments/a2_cap_sensitivity.py`, common random numbers, 262,144 paths):

| | paper | 5x | 8x | 10x | 12x | 15x | 20x | 50x |
|---|---|---|---|---|---|---|---|---|
| baseline, b = 0.25 | 144.1 | 141.1 | 142.7 | 143.2 | 143.5 | 143.8 | 144.0 | 144.4 |
| baseline, b = 0.5 | 202.1 | 177.7 | 192.5 | 198.8 | 203.6 | 209.2 | 216.0 | 234.1 |
| baseline, b = 1 | 323.3 | 228.0 | 287.2 | 317.9 | 344.3 | 378.6 | 426.3 | 616.9 |
| steep, b = 1 | 186.3 | 157.4 | 175.9 | 183.3 | 188.9 | 195.1 | 202.4 | 218.1 |

Between 10x and 12x, gamma(40y) at b = 1 moves 26bp; the whole remaining
gap is 5-8bp. The paper's values correspond to effective caps of about
10.4x (baseline b = 1), 11.1x (steep b = 1) and 11.4x (baseline b = 0.5):
not one level, so a different cap level is not by itself the explanation,
and choosing one to fit would be tuning. What "initial volatility" means
(sigma; sigma_r(0), which at b > 0 involves Y_1(0) / Ybar(0); the local or
the total volatility), whether the cap is on the volatility or on the
variance, and where in the step it is applied all move these numbers by
amounts of this order. The paper's high-skew convexities are therefore
partly properties of the cap; that the shortfall is 1.7-2.6% and grows with
b is consistent with that, but we cannot pin it down further without the
definition.

---

## B3. The Gaussian decomposition example

**Status: fixed.** `examples/gaussian_decomposition.py` now uses the paper's
baseline curve (`markets.paper_curve("baseline")`) instead of an
illustrative Nelson-Siegel curve and compares Monte Carlo, the Appendix D
closed forms and Figure 2. The uncalibrated zeroth-order skew sweep it used
to carry is dropped (`examples/skew_sweep.py` supersedes it).

Figure 2 plots theta and gamma on a right axis that the digitiser did not
read; `tools/digitise_figures.py` now reads them for Figures 2 and 5 (same
scale as the left axis, offset so that gamma(0) = 0), and the overlay check
confirms every point ([Figure 2](investigation/overlay_fig2.png),
[Figure 5](investigation/overlay_fig5.png)). Only the new series were added
to the data file: re-running the digitiser in this environment changes 13
existing values by 0.01bp (numpy rounding), and they were left as they were
so that no committed result moves.

| T | Y_eh: this code / Figure 2 | theta | gamma |
|---|---|---|---|
| 1 | 299.11 / 299.35 | 6.97 / 7.01 | 0.16 / 0.16 |
| 5 | 299.58 / 299.50 | 33.42 / 33.40 | 3.73 / 3.72 |
| 10 | 300.33 / 300.30 | 62.23 / 62.20 | 13.39 / 13.40 |
| 20 | 300.62 / 300.61 | 113.73 / 113.75 | 43.54 / 43.60 |
| 30 | 302.63 / 302.67 | 150.93 / 150.95 | 80.55 / 80.56 |
| 40 | 304.94 / 304.80 | 176.30 / 176.41 | 119.00 / 119.08 |

Monte Carlo against closed form: 0.002bp at most. Closed form against the
figure: 0.24 (Y_eh), 0.12 (theta), 0.08bp (gamma) at most, i.e.
digitisation error. `tests/test_figure_2.py` checks the closed forms against
all four series of Figures 2 and 5 to 0.5bp.
