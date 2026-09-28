#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/data/dev2/src/fa6d6ff3f2e99b17ca273aae4da4581a567d7d6c-src_training_decision2/src/training/decision2
CKPT=/data/dev2/runs/release/inputs/dev2-0p6b/staging-06bm4-62c61c10
C=/data/dev2/runs/release/inputs/dev2-0p6b/cal698
RUN=/data/dev2/runs/release/dev2-0p6b-cal698-rescore
end=$(date -u -d "+20 minutes" +%FT%TZ)
collect() {
  /data/dev2/src/fa6d6ff3f2e99b17ca273aae4da4581a567d7d6c-src_training_decision2/src/training/decision2/v2/eval/run_same_panel.sh --gpu 0 --track release-06b --shared-lease release --src fa6d6ff3f2e99b17ca273aae4da4581a567d7d6c-src_training_decision2 --run-dir "$1"     --model-dir $CKPT --mount $C --purpose "DEV2.0-0.6B CAL698 calibrated re-score" --expected-end "$end"     -- --adapter decision2-typed --model-path $CKPT --revision 0f96aa3932ea501589f794eff9949b52fd1c64835b92f0392da860c22b0442bf     --extra model_id=dev2-06b/m4-t-a7-soup-cal698 --extra calibration=$C/calibration.json --extra max_length=8192 "${@:2}"
}
collect $RUN
cd /data/dev2/src/fa6d6ff3f2e99b17ca273aae4da4581a567d7d6c-src_training_decision2/src/training/decision2
python3 -m v2.eval.same_panel seal --run-dir $RUN
python3 -m v2.eval.same_panel report --run-dir $RUN --label "DEV2.0-0.6B m4-t-a7-soup with CAL698 temperatures (post-key same-panel)" --tier 0.6B --family decision2 --count-safetensors $CKPT
for pair in kai1=/data/dev2/runs/eval/m1-adopt/kai1 kai1-8k=/data/dev2/runs/06b/m2/formal/kai1-native-8k lex=/data/dev2/runs/eval/m1/r3-lex bosun06=/data/dev2/runs/eval/m1-adopt/bosun gliner25=/data/dev2/runs/eval/m1/p1-gliner25; do
  python3 -m v2.eval.same_panel compare --run-dir $RUN --comparator-run-dir "${pair#*=}" --left-name m4-t-a7-soup-cal698 --right-name "${pair%%=*}"
done
collect $RUN-mlx --panels mlx-diag
python3 -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 --predictions $RUN-mlx/output/mlx-diag.predictions.jsonl --output $RUN-mlx/mlx-diag.score.json
echo RESCORE-DONE
