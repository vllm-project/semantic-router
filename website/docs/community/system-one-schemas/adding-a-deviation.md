---
title: Adding a deviation
description: How to declare a tested, documented way a System One backend departs from its schema.
---

# Adding a deviation

A deviation is a named, declared way a backend departs from the schema it
claims, such as `state_text` for a backend that refuses any state but a
string. Add one only for a real non-conformance you have tested against the
backend.

1. **Write the rewriter** in `src/semantic-router/pkg/systemonecodec`: a
   stateless type with `ID` (lower snake case), `AppliesTo` (the schemas it
   applies to), `RewriteRequest`, which edits a planned call and returns what it
   changed as diagnostics, and `RewriteReply`. Deviations run after the codec's
   plan, in the order a deployment lists them.
2. **Register it** in `Builtin()`; the registry refuses a deviation that applies
   to no registered schema, and configuration refuses one that does not apply
   to the deployment's schema.
3. **Test it** on its own: the rewritten request byte for byte, a call it
   leaves alone, and, where it changes what a model reads, the families that
   read the rewritten form the same.
4. **Document it** with a row in the deviations table of
   [System One schemas](../../system-one/use/schemas.md#deviations): the
   schema, what it does and which backends need it.
