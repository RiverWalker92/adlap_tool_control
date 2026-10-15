# Gearbox

From this directory, run `python run_all.py`. Use `--plots-only` to redraw
figures, `--combinations-only` to reuse features, or `--verify-only` to check
saved results. Install dependencies from the parent `requirements.txt`.

`code/analyse.py` extracts features and evaluates the original models.
`code/combinations.py` evaluates 24 singles and 276 pairs and writes CSV tables.
`code/plots.py` generates every current gearbox figure.
`code/verify.py` checks profiles, slopes and saved decisions.

Outputs: PNG in `outputs/figures/`, compact counts in `outputs/tables/`,
full combination results in `outputs/combinations/`. The selected result table
includes 30, 10 and 5 rotations. Only the latter two are cross-validated.

The main RMS figure remains the reference model; the mixed-pair figure shows
three equally successful alternatives in the recorded data. Parameter selection
is exploratory. All degraded blocks originate from one run.
