package shadow

import (
	"encoding/json"
	"fmt"
)

// Evidence packs arm results and the (optional) judge decision into replay
// evidence: a verdict string plus a string->string metadata map. Only opaque
// arm ids and compact features (outcome, latency, usage) are produced — no
// response content and no model identity, per issue #3376 non-goals / D7. The
// opaque->model mapping stays in the deployment config; arm identity lives in
// observability. A zero-value decision means the judge is not configured, in
// which case the arms are still recorded under the "observed" verdict.
func (j *Judge) Evidence(results []ArmResult, decision JudgeDecision) (verdict string, metadata map[string]string) {
	arms := make([]map[string]interface{}, 0, len(results))
	for _, res := range results {
		entry := map[string]interface{}{
			"id":         j.BlindID(res.Arm),
			"outcome":    string(res.Outcome),
			"latency_ms": res.LatencyMS,
		}
		if res.PromptTokens > 0 {
			entry["prompt_tokens"] = res.PromptTokens
		}
		if res.CompletionTokens > 0 {
			entry["completion_tokens"] = res.CompletionTokens
		}
		arms = append(arms, entry)
	}
	armsJSON, _ := json.Marshal(arms)

	metadata = map[string]string{"arms": string(armsJSON)}
	verdict = string(decision.Outcome)
	if verdict == "" {
		verdict = "observed"
		metadata["judge"] = "disabled"
		return verdict, metadata
	}
	metadata["judge_outcome"] = verdict
	metadata["judge_model"] = decision.JudgeModel
	if decision.JudgeRubricVersion != "" {
		metadata["rubric_version"] = decision.JudgeRubricVersion
	}
	if decision.WinnerArmID != "" {
		metadata["winner_arm_id"] = decision.WinnerArmID
	}
	for i, id := range decision.TieArmIDs {
		metadata[fmt.Sprintf("tie_arm_id_%d", i+1)] = id
	}
	return verdict, metadata
}
