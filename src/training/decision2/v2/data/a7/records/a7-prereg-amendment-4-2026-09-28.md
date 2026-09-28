# A7 preregistration amendment 4: Noul balance for A7r (2026-09-28)

Committed before any `a7-rec10-v2` row is built. Version `a7-rec10-v2` = the
rules of amendment 3 plus the balance rule below. `a7-rec10-v1` was built once
(node B, commit `f548fa925`) and is not published.

## Finding in the v1 build (diagnostic)

Rule 7e mapped all 19,075 held rows (0 unmapped, 0 oracle disagreements on the
rubric Score rows). After deduplication (3,703 Stage1 replays inside Stage2) and
admission, the rubric Score cell failed rule 7c (option-only 0.520 and
state-removed 0.505 against a 0.366 majority: the stated level ranges reveal
the likely level) and was dropped as preregistered, leaving Noul only. The Noul
cells pass the shortcut gates at their class prior, but two generators are
imbalanced: transition-set true share 0.16–0.17 and policy 0.40–0.42
(authorization 0.49–0.50). Every other A7 Noul sub-arm has a true share of
0.49–0.51 and data-arms v1 balance Noul to 45–55%; a heavily skewed family
would teach a family-level answer prior.

## Rule (A7r only)

After partition assignment, each Noul (family, language) cell of each part
keeps all rows of its minority class and at most
floor(minority × 0.55 / 0.45) rows of its majority class, chosen in
`sha256("a7r-balance:" + id)` order, so every cell's true share lies in
[0.45, 0.55]. Score rows are unaffected (and the rubric Score cell is still
subject to rule 7c). Counts dropped by this rule are recorded in the build
manifest (`noul_balance`). Everything else is as amendment 3.
