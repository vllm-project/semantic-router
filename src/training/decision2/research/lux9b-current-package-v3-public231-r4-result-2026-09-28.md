# Lux 1.0 current-package v3/public-231 control: r4 result

The separately preregistered r4 **completed** one fresh current-package
Decision 1.0 Lux-9B comparison. It is **post-key same-panel evidence**, since
the project has accessed v3 answers elsewhere. It is neither a Decision 2.0
candidate, a new blind test, nor an official closed JevBench ranking. No r1,
r2 or incomplete r3 prediction was reused.

## Identity and chronology

| Check | Result |
| --- | --- |
| Weight package | `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`; current 29-file bundle SHA-256 `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`; actual deployed parameters 7,940,895,744. All file hashes and HF revision metadata passed. |
| Prospective r4 protocol | Signed preregistration commit `7bf1ac8dc6ad8b371e91a1bb694d1e820c904022`; preregistration SHA-256 `30c73b7c87ff099dffe61a80fd5b13aea97892a6b9cdc5302f26e450540343f5`; private immutable run lock SHA-256 `efc7242d945cbfd69b7f9dcc935b6889d9386b607f2b801d293a88216acbce17`. |
| Native adapter | Published `DecisionModel.decide` with original state/questions; collector SHA-256 `9925fa0d486de6ee2117553ba8f72413df32528b3b66ecc90427f974a1f9f26e`, adapter `native-published-v2-overbudget-invalid-v1`. Only exact native input-limit errors become invalid full originals. No truncation, option removal or chat conversion. |
| Technical dry-run | The preidentified unscored CSS original produced one invalid answer with the exact native over-budget marker. 20.340 GPU-seconds; private receipt SHA-256 `82ecf0bb3774c4db8ea199df4706c31129b6551d8903f1937c2ec3a8e9de3d53`. |
| Complete collection | Fresh typed FINAL 1,600 originals / 2,000 answers; CSS15 6,547 originals; public-231 231 originals. Formal GPU wall time 464.880 seconds; technical plus formal **0.134783 GPU-hours**. All collectors exited successfully. |
| Freeze and scoring | Three-panel gold-free joint seal SHA-256 `6d8e187b2fb7d291e40d8b4b06e599217b97aad2cd2f49267154dfba21f11781` was fsynced before any target/scorer read. Private typed/CSS/public score hashes are `fb96c3c577fd8853c85c43328dc2aea11d51f2deca4a9a90077091e17c4c01e4`, `ea05cc8ee333d209984798a9e4678b4dacf3501e5899106e7555c61b74d93d1d`, and `574e4fc19d46b14eac84166e57027f0f93fcb0ea4b295c9af612a1eb1de62c5f`. Private final DONE receipt SHA-256 `954f2550353b3982fb5d8a4f5bb9d17083a62c61f5025aa58378e99a6404ee67`. |

The native over-budget rule affected **four CSS originals and answer slots**;
all four stayed in the 6,547 denominator and scored invalid/wrong. Typed FINAL
and public-231 each had zero native over-budget or other invalid answers.
One of the 15 CSS tasks contains all four over-budget originals. The GPU was
released with no task container retained.

## Same-panel scores

| Panel / metric | Lux 1.0 current package |
| --- | ---: |
| Typed FINAL `T`, four-family macro accuracy | **0.778125** |
| CSS15 `H`, median task macro-F1 | **0.564370** |
| JevArena v3 `100 × sqrt(T × H)` | **66.268** |
| Typed answer accuracy, all 2,000 slots | 1,645/2,000 = 82.25% |
| Choice / Noul / Score | 712/800 (89.00%) / 705/800 (88.125%) / 228/400 (57.00%) |
| JevBench public-only 231, all items | **183/231 = 79.22%** |
| Public easy / standard / hard | 48/48 / 67/72 / 68/111 |

Typed family accuracies: constraint competition 312/400 (78.00%), evidence
join 800/800 (100.00%), exception stack 305/400 (76.25%), resource ledger
228/400 (57.00%). The typed Score/resource-ledger axis remains much weaker
than Choice and Noul, even though the overall v3 control is substantially
higher than 60.994 reported for the separate JPT-9B peer. The public-231
score is below that peer's 197/231. Treat each as a separate same-panel
comparison with its own native adapter and frozen package, not as proof of a
general ranking or parameter frontier.

CSS15 per-task macro-F1 (%): conv_go_awry 55.51, emotion 53.02, flute 68.40,
ibc 57.24, indian_english_dialect 41.29, media_ideology 56.44, mrf 78.43,
persuasion 53.57, raop 55.47, reddit_humor 59.77, talklife 26.95,
tempowic 62.40, tropes 12.84, wiki_corpus 58.77 and wiki_politeness 56.48.
The four invalid CSS originals are in tropes. The task-level median, rather
than the sample-weighted micro average, defines H.

This is the eligible **Lux1 9B baseline** for a future 2.0 9B same-panel
comparison. A 2.0 model still needs its own qualified frozen package, fresh
complete predictions, overlap/lineage checks and paired interval before a
release claim. The r4 run did not trigger another training arm, HF mutation or
new GPU experiment. Full predictions, protected targets, item scores and
internal paths remain in private receipts; the private aggregate SHA-256 is
`b1909d4b9374f5cd4e7a072f01429f9bb2b2a8a4eb91fd912e98b004be338688`.
