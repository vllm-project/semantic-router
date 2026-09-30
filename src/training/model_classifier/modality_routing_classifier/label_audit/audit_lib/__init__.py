"""Building blocks of the blind label audit, one module per responsibility.

  constants     paths, label codes and limits
  dataset       loading a split, clipping text, pinning the dataset by hash
  judgment      parsing judgment lines and turning them into records
  checkpoint    the append-only judgment store and choosing what to judge next
  stats         kappa, McNemar and accuracy
  human_review  the blinded sheet for a human spot-check and its error estimate
  api_judge     judging through the Claude API
  report        the text report

The command-line tool that wires them together is ../judge_labels.py.
"""
