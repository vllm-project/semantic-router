# Fleet Sim Optimizer Notes

## Scope

- `src/fleet-sim/fleet_sim/optimizer/**`

## Responsibilities

- Keep analytical sizing, DES verification or calibration, power or flexibility analysis, and public export curation on separate seams.
- Treat `base.py` as the ratcheted hotspot for shared optimizer kernels, not as the default home for every new optimizer feature.
- Keep `optimizer/__init__.py` as the optimizer package export seam and coordinate any intentionally root-level re-exports with `fleet_sim/__init__.py`.

## Change Rules

- Do not add new optimizer families, reporting helpers, or public dataclass inventories back into `base.py` if a sibling module can own them.
- When adding a new optimizer capability, decide first whether it belongs to analytical sizing, simulation verification, flexibility or TPW analysis, or export policy; do not mix those concerns by default.
- Keep public export changes deliberate: update `optimizer/__init__.py` first, and only widen `fleet_sim/__init__.py` when the root package truly needs to expose the symbol.
- If a change touches both internal optimizer kernels and package export lists, treat that as a design smell and look for a narrower helper or export seam first.
