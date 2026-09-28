# Sol 2B ShARC v4: corrected private blind packet ready for independent review

The prospective [v4 correction](sol2b-sharc-blind-packet-v4-prereg-2026-09-28.md) ran on publisher TRAIN only, from the exact signed code mirror at `31cc29aa2`. It replaces the [invalidated v3 packet](sol2b-sharc-blind-packet-v3-result-2026-09-28.md) for review; v3 remains as a failed receipt. No one on the packet-building task reviewed the 24 pair answers or policy content. No training rows were admitted, no model ran, and GPU usage was zero.

| Source and privacy check | V4 result |
| --- | ---: |
| Publisher TRAIN rows | 21,890 |
| Publisher negative-question/scenario TRAIN IDs excluded | 4,017 |
| Eligible opposite-label trees / distinct source URLs | 589 / 180 |
| Selected pairs / source URLs / trees / normalized snippets | 24 / 24 / 24 / 24 |
| Hidden Yes-first / No-first state order | 12 / 12 |
| Public blind-ID parity predicts hidden order | **No** |
| Visible state length | 178–990 characters, median 407 |
| Rows admitted / GPU-hours | 0 / 0 |

The v4 builder file SHA-256 is `dc1fd642b38e97a934f107cc84035ce9a990b0a7049484323b9c568b901af2b8`. Its exact local and remote mirror hashes matched. The new private blind packet SHA-256 is `dbef8d7a78d897907534bd47cd4247f88746dd06565e516274d12903ba49ef4e`, the separate private ID mapping SHA-256 is `e503a52113bebe86ab7722f73a4e30133257b0dc5b464068116d8064a7b66fb3`, and the aggregate receipt SHA-256 is `a202e2762cd7faa366bbaec3f0696e498291aac6dda3d271e970ce0beff735a7`. The selected source IDs are identical to v3; only blind state order and defensive history-field copying changed. All files are mode 0600 in a mode 0700 directory and absent from public artifacts.

An automated structural inspection confirmed that reviewer pairs contain only blind ID, rule snippet, question and two states; each state contains only A/B ID, scenario and publisher-visible follow-up history. Publisher `answer`, `evidence`, source URL, tree ID and utterance ID are absent. The private mapping retains exact source URL, tree/utterance IDs and exact/normalized snippet hashes, without labels, for later group-level overlap checks. The reviewer must receive **only the v4 blind packet**, never the mapping, raw publisher TRAIN archive or v3 packet.

**Next gate:** independent blind semantic judgments and separate target adjudication under the unchanged ≥18/24 evidence-necessary floor. Then run source-level exact/near/semantic overlap, rights, native length, state-removed shortcut, matched token budget and zero-step source parity checks before any Sol 2B training arm. The 589 metadata candidates remain **HOLD**, not a validated training set.
