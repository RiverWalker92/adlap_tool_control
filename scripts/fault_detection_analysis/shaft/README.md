# Instrument shaft

From this directory, run `python run_all.py`. Use `--plots-only` to redraw
figures or `--verify-only` to check saved results. Install dependencies from
the parent `requirements.txt`.

`code/shaft_features.py` extracts native-cycle current and video features.
`code/shaft_analysis.py` fits the original 113 candidate feature sets.
`code/plots.py` contains distributions, correlations and all LDA plots.
`code/waveforms.py` makes the full-amplitude waveform and tracking overview.
`code/verify.py` reconstructs all saved scores and counts.

The primary analysis contains 216 individual cycles. Supplementary plots
use 72 three-cycle feature means; their filenames and titles distinguish them.
Training and testing are separated by complete run. Outputs do not aggregate
the individual classifications into a whole-run diagnosis.

PNG files are in `outputs/figures/`; compact CSV tables are in `outputs/tables/`.
All fold predictions, model coefficients and observation metadata remain in
`outputs/`. The model registry is retained in `analysis_metadata.json`.
