# Paired VLA precision--execution study: public analysis release

This repository contains de-identified outcomes from an independent 2 x 2
final grid (BF16 or packed W4/A8; execute 1 or 4 actions),
plus labelled single-condition development diagnostics. The simulator used
eight LIBERO-spatial tasks, 20 initial states per task and three seeds.

## Contents

- `data/episodes.csv`: all 1,920 final-grid outcomes, including failures,
  wall time, action steps and inference-call counts. The pairing key is
  task_id, state_id and seed.
- `data/batches.csv`: 96 batch summaries, including GPU allocator peaks.
- `data/frozen_analysis.json`: frozen aggregate estimates and task-cluster
  bootstrap intervals used in the manuscript.
- `data/table-*.csv`: manuscript table inputs.
- `data/*development*.csv` and the named JSON files: diagnostics selected
  after the independent final grid. These are not confirmation data.
- `code/reproduce_summary.py`: reproducible primary success and all-attempt
  episode-time estimates from `episodes.csv`.
- `code/build_manuscript_figures.py`: regenerates the six manuscript figures
  from the released aggregate and development inputs; requires matplotlib
  and NumPy.
- `figures/figure0.pdf` through `figure5.pdf`: manuscript figure files.
- `SHA256SUMS`: integrity checks for all release files.

Run `python3 code/reproduce_summary.py` from any working directory.
The script prints the arm means and paired task-cluster bootstrap intervals.
Run `python3 code/build_manuscript_figures.py` to regenerate the figures.
The originally frozen analysis used 10,000 cluster draws with seed 20261017;
the released script applies the same resampling convention. Task clusters,
not episodes, are the unit of resampling.

The CSVs are derived from 96 returned result archives. During packaging,
each archive, summary, runner receipt and each internal evidence file was
checked against its recorded SHA-256; only batches with OS child wait 0 were
accepted. Raw action traces, model weights, simulator assets, and restricted
node logs are not included. The frozen aggregate also reports controller
GPU seconds per success; its original controller receipts are retained in
the private project archive and are not reconstructible from the CSVs alone.
The development diagnostics are one-condition mechanism evidence, not an
independent quality estimate. No physical robot or energy measurement is
represented here.
