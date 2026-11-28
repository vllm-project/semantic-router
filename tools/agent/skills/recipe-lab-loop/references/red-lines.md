# Red lines for loop proposals

Goodhart's law is the failure mode of every self-improvement loop: once a
metric becomes the target it stops measuring. These rules are the agent-side
mirror of the contract checks in `recipe-lab/contract.py`.

1. Dataset trisection. `calib` is visible to proposers, `dev` is for their
   own pre-checks, `frozen-test` is touched exactly once by the final gate.
   The same cases must never both drive a proposal and grade it.
2. Guard floors are a veto. Accuracy gains purchased with PII or jailbreak
   recall are recorded as FAIL, always.
3. Holdout quarantine. sr-bench holdout failures are never tuning input
   ("Do not tune on holdout failures").
4. No cross-hash comparisons. A ledger record's `dataset_hash` changes ->
   its numbers are incomparable with earlier rounds.
5. One variable per candidate. If you changed two things and the verdict is
   positive, you cannot say which change caused it; the ledger then records
   an unattributable result.
6. Evidence over conclusions. An interrupted run is INCONCLUSIVE, not FAIL.
   A proposal that would read a failure as "the recipe is worse" is
   manufacturing a conclusion the evidence does not support.
