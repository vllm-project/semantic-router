# Draft proposal snippet

## Counterfactual and state-reduction auditing for Router Replay

This draft proposes a small, implementation-independent audit layer around
Router Replay. The initial PoC evaluates three invariance families:

1. **Local perturbation invariance** — how far a recorded route is from a
   counterfactual decision boundary.
2. **Order invariance** — whether controlled state updates `A→B` and `B→A`
   converge to the same terminal routing state.
3. **Grouping invariance** — whether a state merge/reduction operator `★`
   satisfies `(A★B)★C == A★(B★C)` at the observable routing layer.

The third item does **not** treat ordinary function composition as
non-associative. Here `★` is explicitly a state update followed by a lossy
projection/reduction step (compression, clipping, thresholding, hysteresis,
or another policy-owned state reduction).

The PoC is read-only and does not modify the router hot path. It first requires
baseline reproduction against a Replay-shaped record and a policy snapshot.
Only then does it report counterfactual/order/grouping residuals.

The current order/grouping counterexamples are synthetic stress fixtures. They
demonstrate the evaluation primitive, not an upstream defect. A next step is to
replace those fixtures with live Router Replay captures from agreed E2E
scenarios while preserving the same report contract.
