---
title: Adding an extension field
description: How to add a field to the vllm-sr System One extension set.
---

# Adding an extension field

The extension specification, `config/schemas/systemone/vllm-sr-5.0.extensions.yaml`,
defines every field vllm-sr adds to the standard once: its level, its schema,
the version it appeared in, the families that honour it, and what a standard
target does with it.

1. **Add the entry.** Give it its `level` (request, question, answer, ...),
   `since`, `families`, `schema` and `summary`, and its `standard_target`:
   `strip` when its standard part carries the meaning, `refuse` when a rule
   that needs it must fail to load, `absent` for a field only a vllm-sr server
   sends. Prefer the standard's own form: a field that overlaps a standard one
   is refused, and the generator fails on a field both specifications define
   differently.
2. **Regenerate.** `make systemone-schemas-generate` writes the Go field
   tables, the model runtime's field tables (`vllm_srun/api/systemone_fields.py`),
   the strip list a standard target uses, and the generated extension reference
   and compatibility matrix. `make systemone-schemas-check` fails on a stale
   output.
3. **Implement it** where it is read: the model runtime families that honour
   it, the Router's codecs when the Router sends or reads it, and a new minor
   version of the extension set (`extension_version`) when it is the first
   addition since a release.
4. **Test it** through both listeners and the client codecs: a 5.0 server
   accepts it, a standard target strips or refuses it as declared, and an
   unknown neighbour is still refused.
