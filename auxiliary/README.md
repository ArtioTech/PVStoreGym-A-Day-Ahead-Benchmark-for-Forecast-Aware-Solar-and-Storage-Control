# Auxiliary Assets

This folder is reserved for non-canonical support artifacts that help local research workflows but should not define the public face of the repository.

Typical contents include:

- archived experiment outputs
- exploratory market-data downloads and sample files
- locally cached vendor wheels
- strict-test or scratch processed datasets

The canonical GitHub-facing entrypoints remain:

- `scripts/build_greek_2025_dataset.py`
- `pv_bess_gym/data/builders/greek_2025.py`
- `data/processed/greek_2025_*.csv`
