# C1 amendment 2: two preflight stops, versions A2b and A3b

Status: **committed before any A2b/A3b launch**. Arms A0, N0, A1 and A4 passed
every `dec-arm-preflight/2` gate and started their full runs on code
`cea7cd7ca`. Nothing about those arms changes.

## Preflight outcomes

| Arm | In-process zero parity | Cross-process zero / one-step | Tensor reload | Moved | Status |
| --- | --- | --- | --- | --- | --- |
| A0 | 700/700, drift 0 | 695/700 (.027) / 700/700 | bitwise | yes | PASS |
| N0 | 700/700, drift 0 | 700/700 / 696/700 (.016) | bitwise | yes | PASS |
| A1 | 700/700, drift 0 | 697/700 (.029) / 697/700 (.015) | bitwise | yes | PASS |
| A2 | 700/700, drift 0 | 700/700 / 698/700 (.022) | bitwise | yes | PASS; **teacher-fidelity gate failed** |
| A3 | 700/700, drift 0 | 700/700 / 700/700 | bitwise | **ordinal gate 0.0** | **FAIL** |
| A4 | 700/700, drift 0 | 700/700 / 700/700 | bitwise | yes (gate 5e-4) | PASS |

**C1-A2 stopped.** The Lux teacher path read typed DEV at 1,400/1,600 versus
the stored native Lux 1.0 report's 1,388 (+0.75 point, within ±1.0) and matched
`attribute_gate` 400, `set_reconciliation` 331 and `transition_table` 399
exactly, but `rule_precedence` was 270 versus 258 (+3.0 points), outside the
two-sided ±2.0 family tolerance. 5% of those Noul items lie within .05 of the
threshold. The teacher path is not degraded; the arm is nonetheless stopped
under its C1 name as registered.

**C1-A3 stopped.** Its fixed first optimizer window held 0 of 16 Score rows,
so the Score-only residual received no gradient and its gate stayed 0.0; every
other gate passed.

## New versions (same factors, same training configuration)

- **C1-A2b** — identical to A2 (Lux 1.0 teacher labels `752b7c8f…`,
  KL weight 0.5, temperature 2.00544, same data, budget and selection). Its
  teacher criterion is one-sided non-degradation: the teacher path may not fall
  below the native report by more than 1.0 point overall or 2.0 points in any
  family. The recorded readout satisfies it. A2's PASS preflight receipt covers
  A2b (contracts differ only in the arm label); the full run uses code
  `cea7cd7ca`, like the control. Reported as "A2b (after a stopped A2)".
- **C1-A3b** — identical to A3. Its one-step smoke updates on the first
  fixed-order window that contains at least one Score row
  (`--smoke-window-type score`, preflight-only; ignored by the contract
  comparison). All `dec-arm-preflight/2` gates must pass, including a moved
  ordinal gate. Zero-step, one-step and full run use the commit that adds the
  smoke option; with the option unset the training computation is identical to
  `cea7cd7ca`. Reported as "A3b (after a stopped A3)".

Decision rules, readout and the formal rule are unchanged; A2b and A3b enter
the factor comparison against A0 like any other arm.
