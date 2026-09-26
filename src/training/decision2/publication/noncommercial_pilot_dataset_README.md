# Decision 2.0 noncommercial pilot: metadata only

This folder documents the separate 5,824-row balanced-human pilot and its
frozen selection/calibration splits. **No TRAIN, SELECT or CAL rows are stored
here.** Source TweetEval and CSS text have source-specific research,
noncommercial and/or redistribution conditions. A private repository is still
a third-party upload, so raw rows remain on the authorized research machine.

| Frozen item | Rows | SHA-256 |
| --- | ---: | --- |
| Balanced-human TRAIN | 5,824 | `e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2` |
| CSS pilot SELECT | 600 | `d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38` |
| Hard CAL | 900 | `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf` |
| Original builder manifest | — | `869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f` |
| Eikos 4B r2 research attestation | — | `a0eb728d4c876a8d3e7f2763f94a594998629f8f9ca9edbf10afaeb8d6bb5d91` |

The original builder manifest, Eikos-specific attestation, and source builders
in this folder let authorized researchers inspect derivation and conditions.
The attestation binds exact training run provenance and cannot be reused for
other runs without creating a new statement. Its `rights_conditions` enumerate
all 19 TRAIN source buckets and the SELECT/CAL source buckets. It records that
HatEval is CC BY-NC, SemEval irony is CC BY-NC-SA, NRC emotion/stance are
research-use with separate commercial arrangements, and other TweetEval task
terms vary. It also discloses unresolved original-text rights for CSS
implicit-hate and coarse-discourse, and MultiNLI annotation/derivative terms.

Primary source terms and attribution are linked in the attestation. The source
builders here contain no raw third-party examples or access tokens. This
metadata folder is **not** a license grant and should not be used as a source
of raw training rows. The main dataset directory contains a separately built
rights-clean control with attributable source licenses.
