# Score v6 English direct-LoRA v2: fixed-pair negative result

**Decision: `DO_NOT_ADVANCE`.** The one permitted English r2 selector use is
consumed. The treatment improved over its matched replay control, but gained
**11/192** correct rather than the prospectively required **at least 12/192**.
No checkpoint, seed, threshold, data mix, or scoring rule was changed after
unblinding. Typed DEV, CSS transfer, Chinese held material, JevArena FINAL,
and release panels were not used to select this candidate.

## Frozen execution and custody

This run follows the signed direct-LoRA v2 preregistration and blind-scorer
addendum. The original 27B `BEST368` adapter-plus-base inference fingerprint
was `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
The two fresh-optimizer arms began from its exact adapter and head; the native
32-row BF16 zero-step start gate passed for both arms with 32/32 identical
argmax and zero observed probability drift. Its private parity receipt SHA-256
was `f09802ee3da2921d66949796ab7efdab9da6c6b7dfb5fb48bb472a99e10441fa`.
No merged full-weight initialization entered this experiment.

Both arms completed their sole fixed checkpoint at step 174 with exactly
174 distinct finite optimizer records, **474,124 admitted native tokens**,
2,777 rows and the same parent SELECT/CAL inputs. Each run has one final
checkpoint, and `BEST`, `LATEST`, and `COMPLETE` agree on it. The final native
adapter-plus-base model fingerprints are
`76790ab598835571d63b5f6523ef39fe4ff5b65490eca7ac1a5740ddc964bd4d`
(treatment A) and
`62e42996b57bce85caa376468f583399fcaf16a7a89e8d071cfd22773c35ff49`
(matched control B). The copied immutable run trees matched byte manifests
on the scoring host. The only planned input difference was the frozen
English Score treatment versus token-matched parent Score control; no
optimizer state was inherited.

The original, A, and B each produced 192/192 valid gold-free native Score
answers under one prompt roster, input-token contract, 1,024-token limit,
temperature 1 and adapter-code SHA-256
`a4b0adeb0012e83596c0f633e488e15878eac68ed0f12c2948c0b5c63709ba0d`.
There were no truncations. The A/B prediction payload SHA-256 values were
`06eca2a581114ac423c3a02885f4f3775476a4dfc7311a8d00201bf1dabcc9b8`
and `9e6141d164bf2d16e6caa26081d3d54912e60152c239fa8ab879a05320ef1200`.
The no-key prerequisite seal passed at SHA-256
`604276358a9fd800313520fa9de6509190cb48d2a717a0a95179095bc0ac73f5`.
Only afterward did the one-use scorer create its unblind marker and read the
frozen private key. The private scored report SHA-256 is
`78160b60ccca791be9fcb2d879ea66d6611e1d40b736d69b3694b9244c9cf1f9`.
No raw rows, answer key, item predictions, or private infrastructure are in
this note.

## English r2 selector result

| Model | Correct / 192 | Invalid | All-three-correct groups / 64 | Valid-answer normalized Brier |
| --- | ---: | ---: | ---: | ---: |
| Original `BEST368` | 127 | 0 | 17 | 0.2145 |
| Treatment A | 138 | 0 | 29 | 0.1698 |
| Matched control B | 127 | 0 | 14 | 0.2102 |

| Operation | A / 48 | B / 48 | A minus B |
| --- | ---: | ---: | ---: |
| Allocation caps | 17 | 17 | 0 |
| Inclusive coverage | 34 | 33 | +1 |
| Independent quorum | 39 | 32 | +7 |
| Waiver precedence | 48 | 45 | +3 |

The classwise correct counts show the tradeoff behind the net gain:

| Gold Score level | Original / 64 | A / 64 | B / 64 | A minus B |
| --- | ---: | ---: | ---: | ---: |
| 0 | 62 | 54 | 55 | -1 |
| 1 | 18 | 43 | 20 | +23 |
| 2 | 47 | 41 | 52 | -11 |

The treatment substantially recovered the middle level, while losing
high-level decisions relative to the matched control. This is an aggregate
diagnostic on a consumed selector; any proposed repair needs a new
prospectively frozen training design and independent selector.

The fixed 10,000-replicate stratified **paired 64-group** bootstrap interval
for A minus B accuracy was **[0.015625, 0.104167]**. It is descriptive for
this one frozen English selector. Two operation slices improved, none lost
more than two items, and the paired interval lower bound was positive. The
predeclared **12-correct margin failed by one item**, so the conjunctive
advance rule fails regardless of these other passes.

On parent English SELECT, A's Choice changed from **223 to 230/277** correct
and Noul from **239 to 238/271**. Both stayed within their separate two-point
accuracy retention floors; normalized Brier improved for each, and invalid
counts remained zero. B ended at **227/277 Choice** and **242/271 Noul**.
This selection panel is not an independent transfer or release test.

## Stop and data boundary

No independent DEV/CSS follow-up is allowed for this pair under the signed
protocol. Preserve the original source and both complete final checkpoints,
the one-use seal/score receipts, source hashes and the negative v1 merge
parity receipt as separate evidence. The full v6 TRAIN candidate remains on
hold pending qualified independent Chinese editorial review; the admitted
English-only pilot does not resolve that gate. No v6 revision was uploaded to
the private Hugging Face dataset by this experiment. Any new Score curriculum
or initialization experiment needs a new prospective protocol and independent
selector, not a retest of this r2 key. No Decision 2.0 model release follows
from this result.
