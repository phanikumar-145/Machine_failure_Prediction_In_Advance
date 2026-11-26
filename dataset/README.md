# Datasets

## 1. AI4I 2020 Predictive Maintenance Dataset (primary)

- **Source**: UCI Machine Learning Repository, dataset 601
- **URL**: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **File**: `dataset/raw/ai4i2020.csv` (10,000 rows x 14 columns, real measured/simulated industrial data)
- **Task**: binary classification - `Machine failure` (0 = normal, 1 = failure)
- **Features**: Type (L/M/H), Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min], plus failure-mode flags
  (TWF, HDF, PWF, OSF, RNF - **never used as model features**; they leak the target).

The raw file is already bundled in this archive. If it is ever removed, run
`python -c "from dataset.data_loader import ensure_ai4i; ensure_ai4i()"`
and it will be re-downloaded from UCI (requires internet access).

## 2. NASA C-MAPSS FD001 Turbofan Engine Degradation (time-series)

- **Source**: NASA Prognostics Center of Excellence, CMAPSS data set, subset FD001
- **URL**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- **Files** (bundled in this archive):
  - `dataset/raw/train_FD001.txt` - run-to-failure trajectories of 100 engines (20,631 cycles)
  - `dataset/raw/test_FD001.txt` - test trajectories (last cycle only is scored)
  - `dataset/raw/RUL_FD001.txt` - true remaining useful life for the test engines
- **Format**: whitespace-separated, no header; 26 columns:
  `unit, cycle, op_setting_1..3, sensor_1..21`
- **Task**: chronological analysis + forecasting of sensor degradation trends.

Both datasets are used as published - nothing is fabricated. If a file is missing,
the code raises a clear `FileNotFoundError` instead of silently substituting synthetic data.

## Provenance of this bundle

`ai4i2020.csv` was downloaded from the UCI static URL above; the FD001 files were
obtained from a public mirror of the NASA PCoE repository (file sizes and content
match the official MATLAB-exported text files).
