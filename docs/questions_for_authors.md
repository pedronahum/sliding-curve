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
