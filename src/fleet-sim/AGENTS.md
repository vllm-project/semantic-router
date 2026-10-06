# Fleet simulator

- Preserve the `fleet_sim` public imports and package data when moving code.
- FastAPI routes own transport only; simulation and catalog behavior stays in
  package modules.
- `fleet_sim/__init__.py` and `optimizer/__init__.py` are deliberate public
  export seams. Root-level re-exports require an actual user-facing need.
- Keep analytical sizing, simulation verification, and power/flexibility
  analysis independently testable.
