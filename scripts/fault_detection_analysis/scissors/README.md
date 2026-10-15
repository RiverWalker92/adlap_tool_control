# Scissors

From this directory, run `python run_all.py`. Use `--plots-only` to redraw
figures, `--combinations-only` to reuse features, or `--verify-only` to check
saved results. Install dependencies from the parent `requirements.txt`.

`code/extract_features.py` extracts the current-residual descriptors.
`code/classify_cuts.py` evaluates the original 121 model definitions.
`code/combinations.py` evaluates 68 singles and 2,278 pairs, performs the nested
selection check and exports fitted candidates.
`code/plots.py` writes all current PNG figures and compact CSV tables.
`code/verify.py` checks source features, folds and candidate predictions.
`code/predict_cuts.py` applies a saved model to a feature table.

Each cut is an observation; complete runs are held out for validation. The
primary binary figure uses cuts 2 and 3 and all filing iterations. Separate
figures retain the all-cut/first-cut comparison and the three intermediate
definitions. Three-class figures use closing/opening range for intermediate 4.

Every current plot is in `outputs/figures/`. Compact summaries are in
`outputs/tables/`; full candidate results are in `outputs/combinations/`.
The comparison table contains exact three-class labels and alarm counts.
Only the alarm counts combine intermediate and blunt into one positive class.

Fundamental and second-harmonic amplitudes describe the selected movement
window. They do not denote motor rotation orders. Parameter and cut selection
remain exploratory, with only two healthy runs.
