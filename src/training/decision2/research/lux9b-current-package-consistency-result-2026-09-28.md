# Lux 1.0 current-package consistency: gold-free result

**The new, prospective current-package repeatability gate passed.** This is
technical qualification of one pinned 1.0 control package, not a JevArena
score, model improvement, or correction to the published example. The
[pre-registration](lux9b-current-package-consistency-prereg-2026-09-28.md)
and code were signed in commit `2f796a91b` before GPU use. No typed FINAL,
CSS15, JevBench or answer key was opened.

| Locked identity / check | Result |
| --- | --- |
| Own Lux package | `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8` |
| Current bundle SHA-256 | `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`; all 29 manifest files independently size- and SHA-checked; HF local revision metadata unanimous |
| Native path | Unmodified collector SHA-256 `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`; current qualified image `ce895822...`; both runs attested revision and exact release runtime |
| Fixed public input | Two requests / five Choice, Noul and Score answer slots, derived only from the released *request* fields; prompt SHA-256 `af207bcc78b557e44f9a63357daf5a31b9793968df641842f3112a1f95ed292c` |
| Independent process comparison | Both fresh processes returned both requests and all five slots. Zero category/type changes; maximum numeric or probability drift **0.0**, versus locked repeatability limit `1e-6`: **PASS** |
| Private evidence | First prediction SHA-256 `b65e5e7d5ae52ec2bd198324e644e81cb177472b4c4048d476e40deb5a5fc053`; second `78f02ade59b1d8abda7cea0b8e41432acfce772636ed5b4e9097600579235691`; input lock `7c737da505684b6ee1c7fdbe750f4bd5fa73633053ba3e388b455fb03336ef72`; verification receipt `be088b2f19904dd8ead5017902bb794fdd0fb832c79cec4a29a61d7567f7e63f` |
| Resource use | One idle-confirmed GPU, two fresh processes, 40 and 49 rounded container wall/GPU seconds, total **89 seconds / 0.024722 GPU-hour** below the 216-second cap; GPU idle after exit; no retained containers |

The previous `verify_lux9b_release_example.py` check still **fails** at
`0.021284254 > 0.020000000`. Its expected answers come from a byte-identical
example already listed with an earlier bundle `d968f7e3...`; the current
bundle is `985ade73...` and its release/materials manifests do not bind the
old example. The old failure was never overwritten, its threshold was not
changed, and the new verifier did not read the old answer fields. The two
checks ask different questions: stale example equality versus current package
repeatability.

The historical publication proof reports separate current-bundle CAL/raw-logit
checks. This small run did not repeat those checks or establish byte-identical
parity with the unavailable historical image. It supports a future **separately
frozen, post-key disclosed** Lux same-panel control: complete gold-free v3
predictions and seal must be recorded before scoring. No formal prediction or
ranking was produced here.

Validation: focused new unit test, `make check` for the changed files, and
`make test-training-contracts` passed. Raw prompts, predictions, logs and
machine-specific paths remain private.
