# Experiments

Scripts behind `docs/investigation.md`. Each runs from this directory,
uses fixed seeds and writes its numbers to `results/` (a JSON file, or a
log of its printed table). They are investigations, not tests: most take
minutes and some hours on a CPU.

    cd experiments
    ../.venv/bin/python a9_measure_scan.py

`common.py` holds shared helpers: a simulator of the benchmark yield at
expiry in any candidate measure (P, spot Q, any forward measure), implied-
vol inversion that reports prices below intrinsic as missing (as the
paper's Figure 4 does), and switches for alternative variance schemes,
local-volatility floors/caps and curve extensions. Scripts are named after
the issue they address (`a9_*` for A9, and so on);
`digitisation_overlay.py` checks the digitised figures against the paper's
images (`pdfimages -png paper.pdf figs/f` first).

The investigation these scripts served is closed: every finding is recorded
in `../docs/investigation.md`, and what was put to the paper's authors, with
their answer, is in `../docs/questions_for_authors.md`. The scripts are kept
so that any number in those documents can be reproduced; nothing here is
needed to use the library and nothing here is installed with it.
`digitisation_overlay.py` is the one script that needs the paper's PDF
locally, which is deliberately not in the repository.
