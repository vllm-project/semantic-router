---
title: Adding a schema codec
description: How to add a System One schema, such as a new version of the standard, to the Router's codec registry.
---

# Adding a schema codec

A schema codec speaks one version of the System One API. Adding one, for
example a new release of the standard as `systemone/0.3`, touches no caller:
configuration resolves `schema` through the registry, and the Router's client
and listener paths go through the codec interfaces.

1. **Describe the specification.** Add its file under `config/schemas/systemone`
   and list it in that directory's README. For the standard, write our own
   description of the interface (field names, types and structure, in our own
   words), with `source` and the `source_sha256` you checked it against; never
   copy a document that carries no license. Run
   `make systemone-schemas-drift` to compare the description with the published
   document.
2. **Regenerate the tables.** `make systemone-schemas-generate` records the
   file and its sha256 in `pkg/systemonecodec/zz_generated_specs.go`; the
   registry refuses a codec whose specification does not match.
3. **Write the codec** in `src/semantic-router/pkg/systemonecodec`: a stateless
   type with `ID`, `Spec`, `Capabilities`, `Plan`, `DecodeReply`, `ListingPath`
   and `DecodeListing`. `Plan` decides every wire call of a batch, one batch call
   or one call per request; read and write only the IR (`systemoneprotocol`),
   never another codec's wire form. A schema a listener may serve also gets a
   `ListenerCodec` (see `listener.go`).
4. **Register it** in `Builtin()` (`registry.go`) and add its ID to the
   `systemone` entry of `pkg/apiregistry` (upstream schemas, and
   `ListenerSchemas` when a listener may serve it).
5. **Test it:** golden plans and replies, the refusals at load, its errors and
   listing, and the conformance tests against a mock server built from the
   specification.
6. **Document it:** the [System One schemas](../../system-one/use/schemas.md)
   page and the compatibility matrix, which is generated from the
   specifications.
