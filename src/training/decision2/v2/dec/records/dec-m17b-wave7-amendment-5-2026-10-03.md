# Decoder M17b wave 7, amendment 5: m75 dropped unread; how the amendment-2/3 rules read after the release (2026-10-03 ≈01:35Z)

Written after `4b-LRHxALL` was released (`ce1bdc9d`) and before reading any other single-arm result:
`4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`, `4b-SDML-lrh` and `4b-LHS17IB4-lrq`, as well as `4b-LRHxALL-L2` and
`4b-LRH2`, are unread. Earlier: [amendment 4](dec-m17b-wave7-amendment-4-2026-10-03.md).

- **`4b-LRHxXALL-m75` is dropped unread.** It is ¾ of the previous release plus ¼ of the full-LR soup. Its
  ingredients put it well below the new release, so its read cannot gate. Its lane GPU (node C GPU7) went to the
  user-requested Index submission worker (COORDINATION 09:30). Its staged package stays on nodes C and F.
- **"Qualifies" (amendment 2, for `4b-LRHxQ`)** means a single arm is not significantly below a single-arm release.
  It is therefore read against the run the rule was written for, `AF-4b-LHS17IB4-lrh-bf16`: the 95% upper bound of
  the chain's own bootstrap must be above 0. The release gate is not affected; every candidate is gated against
  `AF-4b-LRHxALL-bf16`'s run (amendment 4).
- **"Single-arm point" (amendment 3's top-3 rule, amendment 4's quarter-LR condition)** is the arm's own IX1 kit
  balanced skill on panel-8.
- `4b-LHS17IB4-lrq`'s chain was restarted on node C GPU5 / 6 with `AF-4b-LRHxALL-bf16` as its reference, so its
  chain bootstrap is the gate. Its shards were untouched, and the restart used the same mirror.
