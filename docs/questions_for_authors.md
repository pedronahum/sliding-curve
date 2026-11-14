# Questions for the authors

The minimal list of questions that remain after the investigation in
`investigation.md`. Each states what we computed, why it does not settle
the point, and what answer would. Numbers are basis points unless stated.
The paper is Piterbarg & Van Der Helm, *Term Premia and Bond Convexity*
(6 Nov 2026).

---

## Q1 (A9)

**How is the "yield option" defined, and are its strikes offsets from the
forward or from the option's own mean?**

Your numbers pin down most of it: the continuously compounded forward yield
`Y(T, T, T+5)` (only it gives your `sigma = 1.1572%` for 82bp; a simple
rate needs 0.93%, a par swap 1.10-1.14%), paid at `T + 5` (at your stated
skew parameters, payment at `T` would put Figure 11 12-14bp higher), with
the implied vol inverted around the mean of the yield in that measure.
What we cannot pin down is where strikes are measured from. Your figures
draw the at-the-money line at the curve forward (387.4 at 20y on the
baseline curve; 446.2 on the market curve), which in that measure lies 25bp
(baseline) and 33bp (market) above the yield's mean. At `sigma = 1.1999%`,
`b = 0.367`, strikes measured from the mean reproduce your Figure 11 fit to
0.7bp; strikes from the forward put the smile 2.4bp high. Figure 8 points
the other way: with strikes from the forward its lowest point falls at the
money, as in your figure; from the mean it falls 50bp above. Which is it,
and is the payment date `T + 5`?

---

## Q2 (A3, A4, A6, A7)

**How many paths (and were they antithetic or quasi-random) did the runs
behind Figures 8, 9 and 11 use?**

At `b = 0`, `eta = 0`, `sigma = 1%` the 20y x 5y option is Gaussian with a
vol of exactly 70.86bp; Figure 8 plots 71.4 and the text says 71.5, and the
eta = 0 line tilts from 71.2 to 71.6. A plain Monte-Carlo run with 4,096
(16,384) paths misses 70.86 by 0.55bp or more 41% (20%) of the time. Your
Figure 8 values at eta = 0.5, 1, 1.5 differ from our precise ones by
-0.7, +1.2 and +0.3bp, non-monotonically. If the runs used a few thousand
paths this also covers Figure 9 at eta = 1.5 (we get 204.2 +- 1.3 against
your 212.8; our single-run spread is 2.6bp at 65,536 paths and would be
5-10bp at 4k-16k), the gap between your skew fit (1.1999%, 0.367) and ours
(1.1890%, 0.353, single-fit spread 0.0055% and 0.0027), and where your
three-parameter fit stopped (A7).

We have since estimated your path count from the scatter of your own
figures (37 points of Figures 3, 6-9, 11 and 12 against precise values;
`investigation.md`, A3). With independent seeds per curve and antithetic
paths, about 10,500 paths (95%: 5,000-20,000) fit every figure, and Figure 9
at eta = 1.5, the eta = 0 level of Figure 8 and the high-skew curves of
Figures 3 and 6 are then all ordinary draws (p = 0.62, 0.87, 0.64). Under
other seedings no single path count fits Figures 8 and 9 together: your
eta = 1.0 smile in Figure 8 sits 1.5-2.0bp above the precise one, while
Figure 9 at eta = 0.5 and 1.0 sits within 0.35bp. So, specifically: how
many paths per figure, were they antithetic, and did the curves of one
figure (the skews of Figures 3 and 6, the etas of Figures 7-9) share a seed?

---

## Q3 (A6, A10)

**What weights did the skew fit of Figure 11 use, and is "sigma = 1.12%" in
Section 9.4 a typo for 1.1999%?**

The text says the fit matches the at-the-money quote exactly; equal weights
miss it by 1.9bp, so we pin it. At `b = 0.367`, 1.1999% (the caption of
Figure 13) reproduces your fitted vols to 0.7bp and Figure 12's gamma(30y)
to 140.5 against 140.3bp; 1.12% misses Figure 11 by 3-5bp under any
reading of the option.

---

## Q4 (A8)

**Could you publish the market curve beyond 30 years and the synthetic
curves beyond 40 years (or the curve nodes)?**

Y_eh and theta at the last plotted maturity depend on forwards up to five
years further out. Reasonable extensions of the plotted curve move Y_eh(30y)
on the market curve from 370.4 to 377.9 (yours: 376.4) and theta by the
same amount the other way; convexity is unaffected. Inside the plotted
range all extensions agree with your figures to 0.3bp.

---

## Q5 (A7)

**In your three-parameter fit, did the objective have an interior minimum
at eta = 0.779?**

Profiling eta (sigma and b refitted at each eta) we find the rms error
falls monotonically, 2.87bp at eta = 0 to 1.30bp at 1.6, and beyond 0.8 by
less than one quote's Monte-Carlo s.e. (about 1.3bp). Your point and ours
(1.80%, 0.36, 1.39) both lie on that valley, along which gamma(30y) runs
from 137 to 464bp (316bp at your point). If your objective had a clear
minimum at 0.779, something in the set-up differs from ours.

---

## Q6 (A1)

**What produced Figure 4, and is it the same pricer as Figures 8 and 11?**

Three independent checks say Figure 4 is not the model of Sections 6 and
9.1 priced as in Figures 8 and 11:

1. The text says at-the-money vol at `b = 1` is about 4bp below `b = 0` at
   the 5y expiry and 6-8bp below at 20y (we get -4.5 and -9.0, or -3.3 and
   -5.1 with forward-referenced strikes); Figure 4 shows 0.0 and +5.9.
2. Figure 4's 20y `b = 0` smile is the option of Figure 8 at `eta = 0`;
   Figure 8 has it flat at 71.4bp, Figure 4 at 68.4bp drooping to 55.1bp at
   +120bp, with no points at +160 and +200bp.
3. At `b = 0` the option is Gaussian in every measure. No measure,
   numeraire, inversion forward or call/put convention gives a Gaussian
   smile that is flat and then falls off a cliff, or loses points half a
   standard deviation from the money.

At 5y our smiles match Figure 4 to 1.1bp; at 20y the `b = 1` slope is
25bp against your 110bp under every scaling and local-vol variant we tried.
If Figure 4 comes from an earlier version of the pricer, the question
closes; if not, the code that produced it would show what differs.

We also tested 96 simple coding errors against the figure, each fixed
before the run and none with a free parameter: the pricing measure, the
discount, the inversion forward, the call/put choice, the strike centre, a
numeraire left in the price, or the form of the local volatility. None
reproduces it, and none even gets both the drooping `b = 0` line and the
steep `b = 1` skew. The two features need forward mismatches of opposite
sign. So if Figure 4 is a bug, it is not a single simple one.

---

## Q7 (A2)

**Exactly how is the volatility cap of Remark 8 defined and applied?**

At high skew the cap binds even without stochastic volatility, and the
convexity depends strongly on it: on the baseline curve at `b = 1`,
gamma(40y) is 287, 318, 344 and 379bp for caps of 8x, 10x, 12x and 15x
(`b = 0.5`: 192, 199, 204, 209bp); without a cap it explodes. With the cap
at 10x `sigma` on the short-rate volatility (local times `sqrt(z)`), applied
at the start of each step, we are 1.7-2.6% below Figures 3 and 6 at
`b >= 0.5` and 40 years (e.g. 317 against 323bp), with Monte-Carlo s.e.
under 1bp and every other setting checked (time step, bootstrap grid,
curve extension, floors). Is "the initial volatility" `sigma`, the
short-rate volatility at time 0, or something else; is the cap on the
volatility or the variance; and is it applied per step to the local part,
to the total, or to the realised path? Also: what expiry grid did the
volatility-scaling bootstrap (81) use? On an annual grid it is not
converged (3.4bp on the steep curve at `b = 1`).

An alternative to a different cap: at 40 years your Figures 3 and 6 lie
above our precise values for every skew, by an amount growing with b
(+0.8, +3.2, +5.4, +6.5bp on the baseline curve). If the curves of a figure
shared one seed, that is what a single run of about 5,000-10,000 paths
looks like (Q2); if they did not, it points to the cap.
