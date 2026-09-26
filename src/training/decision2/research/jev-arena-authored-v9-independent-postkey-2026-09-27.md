# JevArena authored v9 R5 DEV12: sealed blind and post-key review

**Verdict: BLOCK_FOR_RELEASE_BENCH.** This was a prospective private development
pilot, not a release benchmark or a model evaluation. The independent reviewer
sealed an original-item report before seeing the gold-free source deletions,
then sealed a second deletion report before any answer key or proof trace was
opened. The post-key operator compared those fixed reports with the frozen
private targets using the signed source script in commit `eb06010a3`.

| Frozen receipt | SHA-256 |
| --- | --- |
| Original gold-free prompts | `84514f8efa2dab34bf2a6f115c8e895969f7c5caf79c32120d3727e27b8c02ce` |
| Gold-free source deletions | `d66853149a4f6f85bf95cb1c0deacc6a58c7e800625d3e0d9a5cc762ed87ed49` |
| Private targets | `671f958dbd29365d859edfd3ccb1c22e31f5978656335cf9460514da673c9966` |
| Private proof traces | `810db5678277f399f6c6dec574cce0f5e63ca1423492e849f6ac7269ca538425` |
| Sealed original-item blind review | `5c542aa48232ff579defc5f743f6eca5208b5e20509cb8cc36b5034192d4fb22` |
| Sealed source-deletion blind review | `2afa3e7851e26c05365c0d18cf0dd2598664a0a450b48ae89bc89ceb05e7df17` |
| Post-key aggregate JSON | `3ca4004779fdf91482d9f52fc43316498da272c9c3342487dfed5ffc2992e2c1` |

The reviewer independently derived the frozen answer for **12/12** originals:
four Choice, four Noul and four Score. This demonstrates answer-key agreement
for this small pilot only. Of 36 nominated essential-source deletions, **33**
were clean, **two** left weak narrative hints, and **one** failed the clean
deletion test: an approval lead still asserted a signature after its approval
source was removed. That surviving statement could disclose the missing
evidence. The mechanical counterfactual proof did not detect this prose leak.

All four Boolean items in this packet resolve true, with true presented first.
The Choice position audit met its preregistered minimum of three positions
(3/3/4/5, one-based), but twelve items cannot establish shortcut resistance.
The fixed editorial gate rejects the source leak and answer-balance weakness.
No v9 item enters the release-authored set; there is no v9 JevArena score,
FINAL access, or publication claim. The next authoring pass must change the
underlying scenarios and evidence presentation, repair every surviving source
reference, balance outcomes without key-driven salt search, and undergo a new
independent blind review before any release-sized expansion.
