# Rights-clean v1 count erratum

The immutable `rights_clean_v1/rights_clean.manifest.json` with SHA-256
`a4ded3bc13f5dcc9cffbd98899714507f17d29b0f4084cc0319f030a32728bdf`
has a descriptive error in `source_rights`: the `oracle SELECT/CAL` entry says
1,200 rows. The actual frozen output receipts and files each contain 300 rows,
so the combined total is **600**. Their SHA-256 values remain:

| Split | Rows | SHA-256 |
| --- | ---: | --- |
| SELECT | 300 | `1d564becab12717f7883c77131b8e8611a2e495c102a71789a1b1e5c2cf9afd4` |
| CAL | 300 | `35c27a2a16271b7295afa2d65474dbdc7c64742fdce23ba26bb2d3792a0921d6` |

No frozen data or manifest bytes have been changed. A later immutable v2
manifest corrects the descriptive count and adds human annotated holdouts.
