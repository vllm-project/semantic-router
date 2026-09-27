# Decider 4B same-panel peer result

This is a pinned, native-output external comparator selected from Decision
Index 0.2.1 by size before the official-Qwen 4B formal readout. Its historical
Index score was used only to select the comparator. The numbers below come from
new predictions on the exact JevArena v3 and public JevBench panels used for
the Decision 2.0 4B screen. Project formal keys had previously been opened for
other arms, so this is post-key same-panel evidence, not a virgin blind test.

## Identity and prediction seal

- Model: `Mapika/decider-4b`, immutable revision
  `eb5fbdfc9448473ec25e399882912863afbdb70e`; stored tensor count
  4,205,751,296. Native published `system_one` adapter, version
  `native-published-v2`, with the released per-type temperatures.
- Exact native inference adapter SHA-256:
  `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`.
  Model weight SHA-256:
  `ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd`.
  Model configuration SHA-256:
  `fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf`.
- Gold-free input panel SHA-256: typed FINAL
  `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`,
  CSS15 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`,
  public231 `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`.
- One isolated GPU inference run exited 0 after 506.81 seconds, approximately
  0.14078 GPU-hour. Exactly 1,600 typed items, 6,547 CSS items, and 231 public
  items were predicted. All three prediction files were sealed before scoring,
  at 2026-09-27 14:16:47 UTC; private seal SHA-256
  `61584fee85ddf1eccd8cbb86b310579fdb5d95b1cedfacd6af5852d3cae369ac`.
  The seal records no gold or score read before it was written and zero invalid
  answer slots in the prediction schema. Prediction SHA-256s: typed
  `17933a9f9c1a8f27f6c7f9f189dcb62e640d6789595788d1e0281e5e8958148a`,
  CSS `c003e7e0e101e02d1ed7a385327da9bdd25b105c90b5302633b952b5f4b2412d`,
  public `eae7a792af38aafb2922c926608410583d10be42e9dad44fd9e35e1b3a5a414c`.

## Same-panel scores

| Panel | Result | Scope |
| --- | ---: | --- |
| JevArena v3 typed FINAL T | **0.689375** | Four-family macro accuracy; 1,600 items, 2,000/2,000 valid answer slots |
| CSS15 transfer H | **0.555480** | Median task macro-F1; 15 tasks, 6,547/6,547 valid items |
| JevArena v3 combined | **61.881646** | `100 × sqrt(T × H)`; authored v3.1 items excluded |
| Public JevBench subset | **192/231 = 83.12%** | Independent public-only replication, not the upstream sealed benchmark |

Typed FINAL by output type: Choice **750/800 = 93.75%**, Noul
**623/800 = 77.875%**, Score **114/400 = 28.5%**. Four family accuracies:
constraint competition **.955**, evidence join **.960**, exception stack
**.5575**, resource ledger **.285**. Native confidence remains poor for several
transfer tasks, so the combined score does not imply universal calibration.

Public231 tiers: easy **48/48**, standard **71/72**, hard **73/111**. The
hard-tier gap and weak typed Score are useful targets for Decision 2.0. In the
same v3 run, own Nox 1.0 scored **56.470236** and the official-Qwen 4B
candidate scored **53.217876**; both are below this peer. Those are current
same-panel scores, not historical Index rows. Peer size comparison should use
the loaded parameter count, not the `4b` suffix alone. No claim about official
JevBench rank or a protected new blind set follows from these figures.

The fixed 5,000-draw paired v3 bootstrap against own Nox 1.0 gives Decider
minus Nox **+5.411410 points**, 95% interval **[+0.504765, +10.537760]**.
Typed `T` is +.0750 with interval [.0421875, .1065625]; CSS `H` is
+.036434 with interval [−.041712, +.126682]. The CSS axis alone is not
resolved by this interval. Typed groups were resampled within family and CSS
tasks/items were resampled together for each model, with invalid answers kept
in the denominator. This paired comparison remains post-key same-panel
evidence, not an unseen external test. Private paired-report SHA-256:
`f10ec811ac8344e38bdad9de0226c318a5c00bed3b7bf557f8528c6e372de8b5`.

## Reproduction and limitations

Pinned scorer source SHA-256: typed
`d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`,
CSS `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`,
public `aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf`.
Gold SHA-256: typed
`707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e`,
CSS `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4`;
public target SHA-256
`abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f`.
Report SHA-256: typed
`8d1efb41de66c823a8290642fa40c87983501d36289d28a4230a811fdaf52d75`,
CSS `2e9e5b175a0d5b2749ee7b709358dc014b22f9bb71148f3e6cb2e7f195c3a41a`,
public `918976ba0a0b7c78e44112fec5fc210d454ec4d8596234af7e47c28eb1199ff`.
The complete per-task CSS table, per-item predictions and frozen receipts remain
private. This peer is a comparator, not an allowed Decision 2.0 initialization.
