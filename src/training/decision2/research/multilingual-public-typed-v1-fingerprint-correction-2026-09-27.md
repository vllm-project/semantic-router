# Public Chinese/Russian typed DEV: v1 input fingerprint correction

**Status: v1 model outputs quarantined; v2 rebuilt and inference in progress.**
This is an exposed supplementary development diagnostic, not JevArena FINAL or
an independent multilingual release score.

The v1 builder used `json.dumps(sort_keys=True)` when writing prompts. That
recursively reordered native Choice criterion keys after targets had already
committed the original insertion order and input SHA-256. On the frozen 4,052
rows, all 4,052 serialized payload hashes differed from their target hashes;
1,187 Choice questions also changed option order. An empty-prediction scorer
smoke had missed this because the v1 scorer checked input hashes only for rows
with a prediction.

Two full native predictions were collected before the mismatch surfaced:
the standalone Decision 2.0 Eikos clean-v2 4B candidate and published Decision
1.0 Nox 4B each produced 4,052 answers. The v1 scorer rejected the former at
the first input fingerprint mismatch. Both complete prediction files remain
private under SHA-256
`7c3a32925caeb319ef39a5db3580c482b6f1e738fd0111218c824e445d1e29e6`
and
`65a8e568719fe9051ab99cec2878fd12ca7a7e66706ee165ade7744eb33ffc63`
respectively. **No v1 model accuracy is reported or reused.**

The source fix increments the adapter protocol to
`decision2-public-multilingual-typed-dev/2`, preserves JSON insertion order,
and checks every serialized prompt against its target fingerprint and Choice
option order during both build and scoring, including empty predictions. A
regression test uses deliberately nonalphabetic Choice keys and proves that
the former sorted serialization fails. The newly built private v2 panel has
the same 4,052 IDs and the same ten ambiguous Russian MASSIVE exclusions. Its
manifest SHA-256 is
`fea7c433419d76e150e606cdca4c5029830a966b7febc674a5478fb729f94ede`,
prompt SHA-256
`4147c42006a4dfb39331936ce99e7c964235e5b3de7ac4cc8c19b9c87dbdb75a`,
and target SHA-256
`d93a64f96e5ee69cd67cef56274f37251e1e311e37c5f4ecc9226fead92369cb`.
The v2 empty-prediction validation passed: all 4,052 questions were counted
missing/invalid. The two 4B native models are being rerun on v2; any scores
will be separately bound to its new panel and their exact packages.
