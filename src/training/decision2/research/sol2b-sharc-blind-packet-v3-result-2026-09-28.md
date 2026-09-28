# Sol 2B ShARC v3: private 24-pair blind packet ready, source still HOLD

The prospective [v3 protocol](sol2b-sharc-blind-packet-v3-prereg-2026-09-28.md) produced a private answer-blind semantic review packet using **publisher TRAIN only**. A separate reviewer has not yet judged its policy validity. This is no training data admission or model result: **0 accepted rows, 0 GPU-hours**. V1/v2 results retain their original meanings.

| CPU source-selection check | Result |
| --- | ---: |
| Publisher TRAIN utterances | 21,890 |
| TRAIN utterances excluded by publisher negative-question/scenario IDs | 4,017 |
| Eligible opposite-label trees after exclusion | 589 |
| Eligible distinct source URLs | 180 |
| Blind packet pairs | 24 |
| Distinct source URLs / trees / normalized snippets in packet | 24 / 24 / 24 |
| Visible input length over selected states | 178–990 characters; median 407 |

The publisher archive and TRAIN member SHA-256 remain `72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e` and `d37d349758a69644fbf49827b1a0893f6099752b0ccc9c7e29bd315e5228429c`. The signed packet code commit is `4974afca3`, whose builder file SHA-256 is `4433f4a957e4650bdc72c06467b769314536c8683df4fef4a8388b549608f961`. An exact code mirror was verified before the CPU run. The private packet, separate ID mapping and aggregate receipt SHA-256 are `e7794c10bbff4d89c17347c721ee430da0f60b9a31278d465c14e56979fff92b`, `be88e03f2e4bdf16792e50ac63195eb13a1eab49136a69c37a64b9db6abf8e78` and `94355106672611d39638e66caedce504715670c8bd91b6a0a4b200f96bbc6b57`. Private files were created with mode 0600 in a mode 0700 directory and are absent from repository and gist.

An automated schema check found no `answer` or `evidence` target keys in the packet or ID mapping. The blind packet contains only the publisher-visible rule, question, scenario and follow-up history; the private mapping separately retains exact source URLs, tree and utterance IDs, and exact/normalized snippet hashes for source-level overlap audit. Its URLs and snippet text are not reproduced here.

**Next gate:** independent blind review under the frozen rubric, followed by separate publisher-target adjudication. At least 18/24 pairs must prove opposite, unambiguous answers that require the visible policy evidence. Then source-level overlap, rights, native length, state-removed shortcut, matched-budget and zero-step parity gates remain. A metadata count alone cannot unlock Sol 2B training.
