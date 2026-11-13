# Legacy files

These are my original files from the first version of this project, kept
unchanged so the progress is visible. `IRIS.py` and `README.md` were not edited.

To run the old script, it needs `iris.data` in the folder you run it from:

```bash
cd legacy
cp ../data/iris.data .
python IRIS.py
rm iris.data
```

Things I notice today about this first version:

- It splits the data without a random seed, so the accuracy changes every run.
- It uses `k=1` and never tests other values of `k`.
- It reports one accuracy number from one split of 15 test samples.

The new notebooks in `../notebooks/` fix these three points step by step.
