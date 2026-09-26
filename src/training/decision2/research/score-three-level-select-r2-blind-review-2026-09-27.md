# Score three-level SELECT r2: independent blind review

R2 was authored under the prospective amendment
`score-three-level-select-r2-method-2026-09-27.md`, signed before any r2 data.
The source freeze is in `score-three-level-select-r2-freeze-2026-09-27.md`.
R1 stays blocked and its packet and review seals remain unchanged. R2 is a
checkpoint-selection diagnostic only, never training data or a JevArena release
axis.

## Freeze and blind judgment

The frozen r2 packet contains **80 independent groups / 240 related rows**,
20 groups per operation, 64 English and 16 Chinese groups. The 30 protected
prompt roles (43,685 rows, including complete Score v6 TRAIN and r1 packets)
had zero exact and bounded near matches; maximum native token length was
273/1,024. This approximate check does not establish semantic independence.

Gold-free packet SHA-256
`091e3023b84d64131a72b23b90b3eacf837027ed23d58045de801e92a331f683`;
reviewer manifest SHA-256
`ee3045f852478b743d24af9c076590b4cbec0bc60999e6dd8b44f308e55fbf9b`.
The exact builder used at freeze remains the signed source revision with SHA-256
`23e9c055a802a7b90f7b31245f193af370c8cb1158ec48d297b2138b0b0ff7a7`.
Its source-branch copy later received an import-order-only lint correction
(SHA-256 `fa2c6b6bcce2108a5bfe98204194da3845b323d521f1771ea0f9d8d3d93d735a`);
the frozen candidate was not rebuilt.
An independent agent read only these bytes and derived **240/240** answers
from the displayed rules. All **80/80** groups have a complete 0/1/2 triplet
and no scripted group flags. Eight complete English groups (24 rows) and four
Chinese groups (12 rows) received additional direct editorial reading.

The original private reviewer script SHA-256 was
`dcaa46c6efaa4cf6854a956a1898c6c884fd371d3320745a0b00c279d04114fe`.
The public source copy was formatted for repository style (SHA-256
`20bf089473d817533445d911f4a75e72344674d73f2ecb0f9d4fb5b47096057d`);
rerunning it on the exact frozen packet reproduced both row and group
judgment files byte for byte. Neither copy imports the author builder or
oracles.

The reviewer sealed judgments at **2026-09-26 22:29:19 UTC**, after source
freeze and before seeing author targets. Private seal SHA-256
`6b792a93fa2a56d2ceac7268bdcbb72bafc5340b81da05d370a0c24d883bc312`;
row judgment SHA-256
`294e6317e961ee2b1fffc33b6105ae718fb86e0cef020157334287bb62665922`.
The reviewer had not opened the author source, key, private join or formal
labels, and no model had run.

## Shortcut and post-key checks

The r1 single-pool margin defect is absent: every pool takes exactly two raw
margin values and signs within its triplet. A further scan of the sealed
gold-free packet found **zero** groups where any individual allocation field,
coverage endpoint, evidence sheet, or waiver subrecord has three distinct
states across 0/1/2. Quorum signatures stay fixed; raw qualifying-report
count cannot distinguish levels 1 and 2. Packet order is not a label cue.
These checks are bounded to visible fields and do not rule out every semantic
shortcut.

After validating packet, review and seal hashes and chronology, the separate
post-key audit checked all 240 packet inputs against author targets and found
**240/240 answer matches, zero input identity disagreements**. Private
aggregate receipt SHA-256
`c6027d5cd61ee0932dba9832298723e697349de023ee5dee5e145a3f998f0c57`.
The sealed judgments were not edited.

## Scope

**English 192 rows may be used only for a preregistered development
diagnostic. Chinese 48 rows remain held pending qualified independent
language review.** The Chinese spot check above was by an agent and does not
certify native or human editorial quality. The four fixed-format rules repeat;
case themes are decorative, and the waiver set tests an active hold only.
Neither an r2 score nor a model gain exists yet. A matched TRAIN experiment
still needs a separately approved TRAIN corpus, run-specific frozen plan,
parent SELECT retention checks, and its own release gates.
