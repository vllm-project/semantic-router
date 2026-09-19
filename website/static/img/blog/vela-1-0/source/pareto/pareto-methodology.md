# Rankings and size–quality comparisons

## Complete benchmarks

The primary score is **Mean(TaskType)**: average tasks within each task type, then give every type equal weight. **Mean(Task)** gives each task equal weight and is reported separately. Scores are multiplied by 100. The English panel contains all 41 MTEB(eng, v2) tasks across seven types. The audio-only MAEB panel contains all 19 tasks across five types and 134 subset/split slots; slots are first aggregated within their task.

Rankings use the September 17, 2026 dedicated MTEB registry snapshots plus both current Vela releases. English has 186 complete registry models; audio has 62. Incomplete rows (151 English, seven audio) are excluded rather than filled or combined across snapshots. Global ranks include 13 English models without a known positive size. Those models cannot enter a size-filtered rank or scatter plot. Ties share competition rank: one plus the number of strictly higher scores.

The plots show every complete model with a known positive total parameter count. The dashed Pareto line connects nondominated observations: no other eligible model has both no more parameters and no lower score, with one strict improvement. Vela points remain visible even when they are below this line. The adjacent table shows the eight highest-scoring eligible models no larger than the highlighted Vela model, plus Vela itself if it ranks outside those eight. The displayed gap subtracts Vela's score from the highest score in this same size-constrained population. All comparisons use unrounded scores.

Vela sizes count all stored model parameters across text, image and audio: Nano 163,771,288 and Mini 1,361,475,288. Peer sizes use registry-reported totals, which can be rounded. Peers include both single-modality specialists and multimodal encoders. Their scores were reported under differing prompts, model revisions and evaluation protocols; these are snapshot-relative comparisons, not matched reproductions or official leaderboard submissions. There is no combined text–image–audio rank. Neither current Vela model lies on these complete-panel frontiers.

[All ranks](complete-panel-ranks.md) · [Complete vectors, source URLs, exclusions and plotted points](general-rank-data.json) · [Figure metadata](general-rank-figures.json) · [Vela evaluation protocol](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/main/benchmarks/EVALUATION.md)

## Evaluation modes

Nano uses default shared text; Mini uses fixed official per-task MTEB instructions in the English comparison. Both use default shared audio. Mini's historical default English result (58.79 Mean(TaskType), 63.49 Mean(Task)) remains available separately and is not a matched causal control. One declared mode per Vela model is counted in each ranking population.

[Fixed instructions and all 41 scores](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/main/benchmarks/instruction-mode.md) · [API equivalence](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/main/benchmarks/instruction-api-equivalence.json)

## Selected task strengths

The task gallery highlights Nano on IMDb classification and NMSQA audio pair classification, and Mini on Mridingham tonic classification and SIB-FLEURS spoken-topic classification. Highlighted models are shown whether or not they lie on the observed frontier; frontier membership is recomputed from all documented eligible points. A task-level position does not establish overall SOTA, and neither a label nor a displayed highlight removes other eligible points from the frontier calculation.

The task plots retain finite scores with known positive total size, including closed-weight models. Registry observations and separately measured peers retain their individual provenance. A stronger new observation can change the frontier without any change in Vela's own score. Complete benchmark results, including regressions, remain in the evaluation report.

[IMDb and Mridingham points and sources](pareto-new-frontiers.json) · [NMSQA and SIB-FLEURS points and sources](pareto-data.json) · [Other retained task observations (CSV)](pareto-points.csv)

Mini's English comparison uses its complete 41-task official-instruction evaluation. Its audio comparison retains the complete 19-task default-mode evaluation through an exact weight and API bridge. Both retain their actual evaluated artifact identities; no full-panel rerun of the combined release is claimed. Nano's English computation is unchanged. Per-task regressions remain in the complete reports.

SIB-FLEURS measures topic classification of multilingual audio, not language identification. This description follows pinned MTEB 2.21.0 task metadata (`mteb/tasks/classification/multilingual/sibfleurs.py`, SHA-256 `f88cc34acc7414dc84ccf8cbbd2f79456076df126f9acc696bd7f45c9481ad55`), with [dataset revision 186b6117](https://huggingface.co/datasets/mteb/sib-fleurs-multilingual-mini/tree/186b61175fbd77059f769b6bf1110d449a2ff311).

Nano now includes the complete 19-task audio evaluation at 163,771,288 total parameters. Its English computation is unchanged. Mini's optional instructed text mode is identified separately; its audio scores and size are unchanged. The new Nano audio point improves the complete-panel aggregate but individual regressions remain visible, including NMSQA and VehicleSoundClustering. The NMSQA highlight is a task comparison, not a claim of frontier membership. The general complete-panel rankings take priority over selected task plots.

Training exposure: each complete reference point retains the snapshot's original `trainingDatasets`, `trainedOnTasks` and `zeroShotPct` fields in [the general-rank data](general-rank-data.json). These are source-declared dataset/task overlap and zero-shot flags, not an independent audit of training examples or held-out splits. A declared overlap does not by itself establish held-out test leakage; an empty or missing declaration does not establish zero exposure. The peer population is therefore not a verified strictly zero-shot comparison. Reported evaluation protocols are retained, and this disclosure changes no peer inclusion, score, rank, gap or frontier.
