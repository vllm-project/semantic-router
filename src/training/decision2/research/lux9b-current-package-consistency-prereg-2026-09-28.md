# Lux 1.0 current-package consistency: provenance correction and prospective gate

**Locked before the new GPU probe.** This is a gold-free technical check for
the 9B Decision 1.0 control. It does not score JevArena, revise the published
model, or relax the previously frozen numerical limit. The old release-example
gate remains recorded as **failed**: five categories agreed, but the `usage`
Score probability differed by `0.021284254`, beyond `0.020000000`. Fresh
process isolation produced zero change; a single-question batch-shape change
was only `0.000953389` and still failed the old gate.

## Why that old comparison is not a current-weight identity test

An earlier Lux release manifest (SHA-256 `bf459fc18212be8e1829b2ed1298bcba89266905164eebcd4d228faddc09ed77`)
binds bundle `d968f7e3f1b8e8302909da4631304cd939f721d3109f49f95591b4af1852e67f`
and lists `model-card-example.json` SHA-256
`12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18`.
The pinned current Lux revision `bd45a30aee8c84032791c245c70f86dee5389cc8`
binds a different bundle,
`985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`.
The example file is **byte-identical** to the earlier example, but the current
release manifest (SHA-256 `f786ce80a159c716f22b7d9c652af765b06f9db083bf43a8b08e67c3be3e69c5`)
and its materials manifest do not list it. The reading-update stage builder
(SHA-256 `3ac30d94c83a3d3f34bfe530e79bc2deee0f162bacc69df42568634cbe9c63fc`)
copies the new bundle plus an enumerated docs set; the previous example was
not regenerated or bound to the new bundle. Its *requests* are fixed public
inputs, but its *answers* are not a valid numerical reference for the new
weights. This is a provenance error in our attempted check, not a pass of that
check or proof that every numerical kernel is unchanged.

The historical reading-update proof used a runtime image with SHA-256 prefix
`a86b65fb`; our current qualified-image probe used prefix `ce895822`. Both
report the same qualified library/profile versions. The original image is not
available on the two authorized nodes, so there is no byte-identical image
contrast in this experiment. Do not attribute the probability difference to
that image change without a controlled rerun.

## Fixed new gate

- Download the complete current own Lux package with the HF CLI at the pinned
  revision. Require the current bundle and release-manifest SHA-256 values
  above and HF local revision metadata. Native loading must verify every
  bundle file. Use the same exact qualified image digest `ce895822...` and
  unmodified native collector SHA-256
  `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`.
- The new verifier source SHA-256 is
  `37abe239c5b48c335e980539180c1e4f47db9f29161a06dc63bc6faf08426ee8`.
  It derives two requests/five Choice, Noul and Score slots from the
  unchanged example **request** fields and never reads its recorded answers.
  Freeze the generated prompt SHA-256 before GPU use.
- Run exactly two independent fresh native processes on one live-confirmed idle
  GPU, each processing both requests in the same order. Require the exact
  current package revision, five complete valid answers, zero category/type
  changes, a release-qualified runtime in both, and maximum absolute answer
  numeric drift at most `1e-6` across processes. This `1e-6` is a new,
  prospective **repeatability** limit for the current bundle, not a revision
  of the old `0.02` example-reference limit. A crash, missing answer, failed
  package attestation, unqualified runtime or drift above limit is a failure.
- The two GPU launches together are capped at 216 wall/GPU seconds (0.06
  GPU-hour); no retry or alternative image, model, prompt, threshold or
  checkpoint. Do not use typed FINAL, CSS15, JevBench or any answer key. Keep
  raw requests, predictions, logs and machine paths private.

Passing this check establishes only reproducibility of the pinned current
package under one qualified native path. A future Lux v3 control would need
its own complete gold-free prediction freeze and post-key disclosure. The
historical publication proof's CAL/raw-logit checks remain separate evidence;
this small gate does not repeat them.
