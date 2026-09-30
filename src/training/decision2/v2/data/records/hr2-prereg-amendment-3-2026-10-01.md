# HR2 preregistration — amendment 3: leak-guard drop before upload (2026-10-01)

Committed after the final build and the blind review, **before any upload**. The review verdict, the gold-error
drops and every other rule are unchanged.

The shared leak guard (`v2/common/check_no_private.sh`), run over the assembled upload tree as for every data
upload, reports 28 findings in 27 TRAIN rows of upstream text: one HelpSteer3 helpfulness row contains a
GitHub-token-like string (twice), and 26 rows contain IPv4 addresses. None is a node address or a secret value
(checked against the nodes and secrets files); the card, manifests and receipts are clean.

**Rule:** every TRAIN or DEV row with a leak-guard finding is dropped, whatever its label, and recorded as
`leak_guard` in `final.json`. `finalize` re-balances by downsampling as before, so the final files only lose rows.
The upload tree is assembled again and must pass the leak guard with no finding.
