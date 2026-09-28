# Decision 2.0 9B + CLM track

Frozen-backbone feature extraction, CLM-style dual-projection readouts and
ordinary Decision-head controls for the 9B tier. Records (protocols,
preregistrations, results) are in `records/`.

Run modules from `src/training/decision2` with this directory on the path:

```bash
cd src/training/decision2
PYTHONPATH=.:v2/9b python -m clm9b.extract --help
python -m unittest discover -s v2/9b/tests -t v2/9b
```

Remote nodes run only exact mirrors of pushed commits inside the pinned
trainer image; see `records/` for the frozen commands.
