# introToFin

This is a project in ECON 1501 Introduction to Fintech.

## Quick Guide

1. Load your data `stock_example.csv`, search all the default data paths `C:\\CODES\\CODE_PYTHON\\stock_sample.csv` in this repository and replace them with your own data path.
2. Run `mlp_v2.py`(MLP), `lambdarank.py`(Lambdarank) or `lgbmReg.py`(GBDT) to get some results like `output_lgbm.csv`, and then check its performance in the `__main__` module of `PortAnalysis.py`.
3. `genetic.py` shows how we find effective factors with genetic programming. `genetic_fac_demo.py` shows our best results from experiments.
   
`Factory.py`, `AlphaMiner_v2.py`, `PortAnalysis.py` are useful tools for data cleaning, single factor evaluation, portfolio strategy construction and backtesting performance assessment.