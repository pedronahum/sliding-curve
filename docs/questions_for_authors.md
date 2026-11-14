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

## Q3 (A6, A10)

**What weights did the skew fit of Figure 11 use, and is "sigma = 1.12%" in
Section 9.4 a typo for 1.1999%?**

The text says the fit matches the at-the-money quote exactly; equal weights
miss it by 1.9bp, so we pin it. At `b = 0.367`, 1.1999% (the caption of
Figure 13) reproduces your fitted vols to 0.7bp and Figure 12's gamma(30y)
to 140.5 against 140.3bp; 1.12% misses Figure 11 by 3-5bp under any
reading of the option.
