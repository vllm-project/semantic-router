# Score curriculum v1: independent gold-blind editorial review

**Decision: BLOCK_FOR_TRAINING.** An independent reviewer verified the frozen
144-row, 48-group TRAIN-only packet and sealed a per-row and per-group verdict
without opening the answer key, proof traces, authoring source or model output.
The packet SHA-256 is
`348bb87547000f5bbe0636209666f9bf80e437343f17f08bea69e196e884169c`;
the sealed private verdict SHA-256 is
`91ed5b7ffc38000e5f29eedb92c46eab9cf77929a82f335b24e819b87764945d`.
The reviewed source TRAIN SHA-256 remains
`79ae47a058d88a78dad45ff0c0dc5391386c6102e55a2202adc06134156e66f7`.

The reviewer found all **144/144** rows uniquely solvable from the visible
rules and state, with no hard ambiguity. All **48/48** three-row groups span
the three ordinal levels. Those positive checks do not establish training
quality: the review packet's source IDs end in the answer level, disclosing
**144/144** keys to a reader. This is a blind-review packet defect. The model
input does not include those IDs, so that leak alone is not a model shortcut.

Two independent shortcuts **are** in the model-facing state. All 36 reviewed
route-depth rows have four, five or six listed links for levels 0, 1 or 2
respectively; link count predicts **36/36** answers without graph traversal.
For timely streaks, the number of on-time days predicts **32/36** answers
without ordering the days. The packet also reuses eight instruction/option
sets and has thin contextual variety. These findings block the intended
reasoning curriculum even if the review IDs were hidden.

The v1 dataset and verdict remain immutable. No post-key approval, GPU
optimizer step, DEV/transfer evaluation, HF dataset update or model release
follows. A new version must remove the ID suffix from the reviewer-visible
packet using a private join, balance irrelevant link and on-time counts across
Score levels in the actual state, repeat mechanical overlap/oracle checks, and
undergo a newly sealed independent blind review. Relabelling this packet is
insufficient.
