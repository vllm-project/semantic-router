# JevArena authored v11 DEV12: frozen gold-free editorial packet

**Status: BLOCK_FOR_RELEASE_BENCH after independent two-stage blind review.**
This is not a release benchmark, training set, model score, or claim of model
improvement. The [prospective method](jev-arena-authored-v11-prereg-2026-09-27.md)
was mirrored into source commit `60dd1e188` before any v11 case text or gold
was authored; the central gist preregistration is `eecf9dece`, SHA-256
`ca6b1053db5182d65154d823eab7c155137e6f511eab30366ff358b8dc93a1e4`.
Signed builder commits are `1855c4b39` and `ff5a8173b`. Construction ran on
the second authorized experiment host using CPU only. The frozen reviewer
receipt contains content hashes and no host identity.

| Frozen item | SHA-256 |
| --- | --- |
| Author specification commitment | `f18420bb74852aa3d807e1513ee73e075018979ea50a75e4ae4ea55b62f5032c` |
| Single private salt commitment | `8ce975488f1a6aafa8c50ed226677238099754dc71b910df1db9d0d6bb923850` |
| Signed builder file | `ae538f1024c7859207d83b8d267a7c6306dc123be882c8bbe85cf9fa91d35d6a` |
| Gold-free original prompts | `94b9ff5db3735ca59e0197700a7a4a2e16b9fd8d045cbfc87ffeb0a8ef6187d5` |
| Gold-free source deletions | `7a47e0ce13e005e9ac1ad1e7c9271ce1e72026840250ff74f8b9fae9a8dde31d` |
| Gold-free reviewer manifest | `9a663ac620795c96fbfc66cea18b1b306d1adc8f40124e23b2c78e0fa6793a93` |
| Gold-free freeze receipt | `267f17f13006911b8a258b217a465ef76dface378bee8d9294b856eda12e049b` |
| Private targets | `7e576f2f7c3826d72bb0cbbeac28523adba280a7c94927025af079d14f8aeb33` |
| Private proof traces | `7eb0bafa4811843363d7aef89f83405d50999a6ba199d1682d439aa8c2271a70` |

The packet has **12 new domains and 12 new semantic operations**, with four
items each for Choice, Noul and Score. The 39 model-visible evidence sources
use eight formats: correspondence, log, minutes, form, letter, schedule,
checklist and report. Every source has a private two-completion witness that
changes the original answer when that source is absent. The renderer deletes
the whole source while preserving the source-neutral task lead, current rule,
other source blocks and question verbatim; all 39 deleted claim spans are
absent from their reduced packets. The program uses two independent rule
oracles. A second build from the same signed source, specification and single
salt matched all hashes byte for byte.

Balance and length gates: Boolean answers 2 true/2 false; Choice answers in
four different displayed positions; Score answers cover four levels. Case
lengths sorted are 210, 239, 256, 257, 264, 278, 283, 317, 347, 510, 542
and 648 words, giving 3 compact, 6 medium and 3 long under the source's
<250 / 250–499 / 500–900 reporting bands. Source prose has no repeated
eight-word sequence, and maximum pairwise trigram Jaccard is 0.029. A
gold-free cross-version check found no exact v9/v10 state duplicate;
maximum five-gram Jaccard was 0.00156 versus v9 and 0.00298 versus v10.

Earlier author-side r1 and r2 drafts are retained privately; neither was
given to a blind reviewer. r1 triggered a quality review because a planned
long case fell just below the useful-length target. r2 met the target, then
the author replaced repetitive source disclaimers with concrete timing,
measurement and process detail before freezing r3. These prospective
changes kept the same facts and one original salt; they were made without
blind judgments or model scores.

Mechanical necessity and surface overlap checks did not establish natural
answerability or absence of subtle prose cues. The independent reviewer
sealed original judgments before seeing deletion variants, then sealed the
deletion judgments before private targets and proofs were opened. The
[post-key audit](jev-arena-authored-v11-independent-postkey-2026-09-27.md)
matched all twelve original targets but found four materially ambiguous
source deletions and additional editorial weaknesses. This frozen DEV packet
is blocked from release use. No FINAL, training, GPU inference or publication
occurred.
