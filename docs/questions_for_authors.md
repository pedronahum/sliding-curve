# Questions for the authors

Three questions remain after the investigation in `investigation.md`; the
earlier seven are merged into them, because four of the old ones (the strike
reference, the fit weights, the three-parameter fit, the curve beyond the
plotted range) have residuals that are bounded and travel together as one
request, and the old cap question is answered or made moot by the old path
count question. Each states what we computed, why it does not settle the
point, and what answer would. Numbers are basis points unless stated. The
paper is Piterbarg & Van Der Helm, *Term Premia and Bond Convexity*
(6 Nov 2026). Publishing the code behind the figures would answer all three
at once.

---

## Q1 (A1, D). What does Figure 4 plot, and which code produced it?

Three checks say Figure 4 is not the model of Sections 6 and 9.1 priced as
Figures 8 and 11: your text puts the at-the-money vol at `b = 1` about 4bp
(5y) and 6-8bp (20y) below `b = 0`, the figure shows 0.0 and +5.9; the
figure's 20y `b = 0` smile is Figure 8's `eta = 0` option, which Figure 8
has flat at 71.4 while Figure 4 has it drooping from 68.4 to 55.1 and losing
the +160 and +200bp points; and at `b = 0` the option is Gaussian in every
measure, so no convention produces that shape - nor does sampling noise
through the floor-clipped inversion, at 4,096-16,384 paths over 64 seeds (D).

96 zero-parameter coding-error hypotheses fail, and three further mechanisms
fail (D): the volatility cap (removing it moves the 20y `b = 1` slope from
39.6 to 39.5, against your 109.3), the wing noise, and a skew compounded per
benchmark period, `b_eff = b T / delta` (it reaches the slope but not the
level: 44.6 and 28.8 at 20y against your 109.3 and 74.3).

The fingerprint is this: your `b = 1` slope between -80 and +200bp is 1.01,
2.00, 3.01 and 4.28 times ours at 5, 10, 15 and 20 years - a factor of
`T / delta` - while the at-the-money levels agree with ours to about 1bp and
the 5y panel matches outright. No local volatility `sigma g(Y_1 / Ybar)` can
carry that factor in its slope at your at-the-money levels: `g'(1) = 4.3`
with `sd(Y_1) / Ybar = 0.64` at 20y forces `rms(g)` above 1.2, where yours
is 1.09. So: what does Figure 4 plot, and does its skew acquire a factor of
the expiry anywhere in the code that produced it? If it comes from an
earlier version of the pricer, the question closes.

---

## Q2 (A2, A3, A4, A6, A7). How many paths, which random numbers, and exactly what cap?

One answer decides whether the 1.7-2.6% (skew) and 4% (`eta = 1.5`) convexity
gaps at the longest maturities are your sampling noise or our cap.

On the noise: at `b = 0`, `eta = 0` the 20y x 5y option is Gaussian with a
vol of exactly 70.86bp; Figure 8 plots 71.4 and its `eta = 0` line tilts.
The scatter of 37 points of Figures 3, 6-9, 11 and 12 against precise values
fits about 10,500 antithetic paths with an independent seed per curve (95%:
5,000-20,000), under which Figure 9 at `eta = 1.5` (we get 204.2 +- 1.3
against your 212.8), the `eta = 0` level of Figure 8 and the high-skew curves
of Figures 3 and 6 (yours above ours by +0.8 to +6.5, growing with `b`) are
all ordinary draws. That estimate assumes pseudo-random draws: with
scrambled Sobol points and a Brownian bridge the same scatter needs only a
few hundred paths, and then the gaps are not noise.

On the cap (Remark 8): on the baseline curve at `b = 1`, gamma(40y) is 287,
318, 344 and 379 for caps of 8x, 10x, 12x and 15x; your 323.3 implies about
10.3x, but your 202.1 at `b = 0.5` implies about 11.6x and your 144.1 at
`b = 0.25` about 20x - no single level fits. Our side is pinned (quasi-random
317.02 +- 0.15; daily steps within 0.5bp; a literal Euler drift or a
predictor-corrector within 1bp; bootstrap grid, curve extension and vol floor
checked).

So, specifically: how many paths per figure, antithetic or quasi-random, and
did the curves of one figure share a seed; is the cap on the volatility or
the variance, on the local part or the total, per step or per path, and
relative to which "initial volatility"; and what expiry grid did the
volatility-scaling bootstrap (81) use (on an annual grid it is not converged:
3.4bp on the steep curve at `b = 1`)?

---

## Q3 (A6, A7, A8, A9). The exact contract, the fit objective, and the curve nodes

The residuals here are bounded, so this is one request rather than four
questions: publish the specification, or the code, of the following.

- **The yield option.** Your numbers pin most of it (the continuously
  compounded forward yield `Y(T, T, T+5)`, paid at `T + 5`, vols inverted
  around the mean of the yield in that measure). What remains is whether
  strikes are offsets from that mean or from the curve forward - Figure 11
  favours the mean (0.7bp), Figure 8 the forward - and confirmation of the
  payment date.
- **The fit weights and path count.** Equal weights miss the at-the-money
  quote by 1.9bp, so we pin it; our fit lands at (1.1905%, 0.353) noise-free
  where yours is (1.1999%, 0.367), and the difference is half a basis point
  of pricing noise in one Monte-Carlo run (gamma(30y) 135.1 against 140.3).
- **The three-parameter fit.** Profiling `eta` (sigma and b refitted at each
  value) the rms error falls monotonically, 2.87bp at `eta = 0` to 1.30bp at
  1.6, and beyond 0.8 by less than one quote's s.e.; your point and ours lie
  on that valley. Did your objective have an interior minimum at
  `eta = 0.779`?
- **The curves beyond the plotted range.** `Y_eh` and `theta` at the last
  plotted maturity depend on forwards up to five years further out;
  reasonable extensions move `Y_eh(30y)` on the market curve between 370.4
  and 377.9 (yours 376.4), and theta the same the other way. Convexity is
  unaffected, and inside the plotted range all extensions agree with your
  figures to 0.3bp.
