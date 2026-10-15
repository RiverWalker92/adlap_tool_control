# Mechanical fault analyses

Current gearbox, instrument-shaft and scissor analyses, with one plotting style.

## Run

Install the dependencies once from this directory:

```text
python -m pip install -r requirements.txt
```

Run every analysis:

```text
python run_all.py
```

Run one experiment:

```text
python run_all.py --project gearbox
python run_all.py --project shaft
python run_all.py --project scissors
```

Regenerate plots from existing numerical results:

```text
python run_all.py --plots-only
```

Check existing results:

```text
python run_all.py --verify-only
```

Each experiment also has its own `run_all.py` with the same modes. Gearbox and
scissors support `--combinations-only` to reuse extracted features. Keep the
`common/` directory beside the three experiment directories.

## Layout

```text
common/                 shared figure style and two-feature LDA implementation
gearbox/
shaft/
scissors/
    run_all.py          entry point for that experiment
    code/               numerical analysis, plots and verification
    data/               supplied measurements and run metadata
    outputs/
        figures/        current PNG figures and their index
        tables/         compact CSV result tables
```

Detailed feature tables, predictions, fold manifests and fitted-model arrays
remain in each experiment's `outputs/` directory. Combination analyses are
under `outputs/combinations/`. JSON and NPZ files retain the information needed
to reproduce decisions and redraw fitted boundaries; they are not report files.

The outputs contain PNG figures and numerical CSV/JSON/NPZ/model files.
The runners do not generate LaTeX, PDF or SVG files. Old report exports,
duplicate image directories, backups and one-off patch scripts are omitted.

## Plot conventions

- Healthy: blue. Degraded or after filing: orange. Intermediate: yellow.
- Different runs use different markers; run names in legends omit timestamps.
- LDA boundaries: dashed grey. Background colours indicate predicted classes.
- Each LDA figure has one legend combining the class/run keys and boundary.
- Parameter and correlation figures use the same serif type and grid style.
- Cut-order figures use separate muted colours for filing iterations because
  the individual iterations, rather than their class labels, are compared.

The plot definitions and numerical models remain separate. Full-data LDA
boundaries are descriptive; validation counts come from the held-out fits.

## Main figures and tables

| Experiment | Figure | Matching table |
|---|---|---|
| Gearbox | `lda_rms.png` | `selected_lda_counts.csv` |
| Gearbox | `lda_mixed_parameters.png` | `selected_lda_counts.csv` |
| Shaft, one cycle | `03_pooled_current_lda.png` | `main_one_cycle_counts.csv` |
| Shaft, three-cycle means | `lda_three_cycle.png` | `supplementary_three_cycle_counts.csv` |
| Scissors, cuts 2 and 3 | `scissors_binary_cuts23.png` | `binary_selected_counts.csv` |
| Scissors, all cuts | `scissors_severity_selections.png` | `severity_cut_selection_comparison.csv` |
| Scissors, cuts 2 and 3 | `scissors_severity_selections_cuts23.png` | `severity_cut_selection_comparison.csv` |

The shaft current-pair figure uses M1 absolute peak and M2 RMS (`current_plane`
in the numerical model registry). The main result table also contains separate
single-parameter and current/video models; their counts belong to their own rows.

The scissor severity comparisons use the same input pairs for both cut
selections: M3 closing signed maximum with opening fundamental amplitude;
M3 closing signed maximum with opening mean; and closing with opening range.
Models are fitted separately for all cuts and cuts 2 and 3. Exact-class counts
and fault-detection counts are both retained in the comparison CSV.

Gearbox tables retain 30-, 10- and 5-rotation outcomes. Thirty-rotation outcomes
are descriptive; ten- and five-rotation counts are repeated validation decisions.
Scissor cut selections and intermediate definitions, and shaft one-/three-cycle
observations, remain separate analyses.

`figure_index.csv` lists all retained figures. `data_inventory.csv` records the
source file hashes. `validation.json` records the cleanup checks.
