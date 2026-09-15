package shadow

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func evidenceWinnerFixtures() (*Judge, []ArmResult, JudgeDecision) {
	arms := []config.ShadowArmConfig{
		{Name: "arm-a", Model: "model-a"},
		{Name: "arm-b", Model: "model-b"},
	}
	judge := NewJudge(config.ShadowJudgeConfig{Model: "judge-m", RubricVersion: "v1"}, arms)
	results := []ArmResult{
		{Arm: "arm-a", Model: "model-a", Outcome: OutcomeCompleted, LatencyMS: 123, PromptTokens: 9, CompletionTokens: 5},
		{Arm: "arm-b", Model: "model-b", Outcome: OutcomeFailed, Err: "boom"},
	}
	decision := JudgeDecision{Outcome: JudgeWinner, WinnerArmID: "arm-1", JudgeModel: "judge-m", JudgeRubricVersion: "v1"}
	return judge, results, decision
}

func TestEvidenceWinnerVerdict(t *testing.T) {
	judge, results, decision := evidenceWinnerFixtures()
	verdict, metadata := judge.Evidence(results, decision)
	if verdict != "winner" {
		t.Fatalf("verdict wrong: %s", verdict)
	}
	if metadata["winner_arm_id"] != "arm-1" {
		t.Fatalf("winner arm id wrong: %s", metadata["winner_arm_id"])
	}
}

func TestEvidenceArmsCompactJSON(t *testing.T) {
	judge, results, _ := evidenceWinnerFixtures()
	_, metadata := judge.Evidence(results, JudgeDecision{Outcome: JudgeWinner, WinnerArmID: "arm-1"})

	var entries []map[string]interface{}
	if err := json.Unmarshal([]byte(metadata["arms"]), &entries); err != nil {
		t.Fatalf("arms not valid json: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("want 2 arms, got %d", len(entries))
	}
	if entries[0]["id"] != "arm-1" || entries[0]["outcome"] != "completed" || entries[0]["latency_ms"] != float64(123) {
		t.Fatalf("first arm entry wrong: %v", entries[0])
	}
	if entries[0]["prompt_tokens"] != float64(9) || entries[0]["completion_tokens"] != float64(5) {
		t.Fatalf("usage not captured: %v", entries[0])
	}
	if entries[1]["outcome"] != "failed" {
		t.Fatalf("failed arm not recorded: %v", entries[1])
	}
}

func TestEvidenceLeaksIdentity(t *testing.T) {
	judge, results, decision := evidenceWinnerFixtures()
	_, metadata := judge.Evidence(results, decision)
	raw, _ := json.Marshal(metadata["arms"] + metadata["judge_outcome"] + metadata["winner_arm_id"])
	for _, leak := range []string{"model-a", "model-b", "arm-a", "arm-b"} {
		if strings.Contains(string(raw), leak) {
			t.Fatalf("evidence leaks %q: %v", leak, metadata)
		}
	}
}

func TestEvidenceJudgeDisabled(t *testing.T) {
	arms := []config.ShadowArmConfig{{Name: "arm-a", Model: "model-a"}}
	judge := NewJudge(config.ShadowJudgeConfig{}, arms)
	results := []ArmResult{{Arm: "arm-a", Model: "model-a", Outcome: OutcomeCompleted}}

	verdict, metadata := judge.Evidence(results, JudgeDecision{})
	if verdict != "observed" || metadata["judge"] != "disabled" {
		t.Fatalf("judge-disabled verdict wrong: %s meta=%v", verdict, metadata)
	}
	if metadata["arms"] == "" {
		t.Fatal("arms not recorded when judge disabled")
	}
}

func TestEvidenceTie(t *testing.T) {
	arms := []config.ShadowArmConfig{{Name: "a", Model: "m"}, {Name: "b", Model: "m"}}
	judge := NewJudge(config.ShadowJudgeConfig{}, arms)
	results := []ArmResult{
		{Arm: "a", Outcome: OutcomeCompleted},
		{Arm: "b", Outcome: OutcomeCompleted},
	}
	decision := JudgeDecision{Outcome: JudgeTie, TieArmIDs: []string{"arm-1", "arm-2"}}

	verdict, metadata := judge.Evidence(results, decision)
	if verdict != "tie" {
		t.Fatalf("verdict wrong: %s", verdict)
	}
	if metadata["tie_arm_id_1"] != "arm-1" || metadata["tie_arm_id_2"] != "arm-2" {
		t.Fatalf("tie ids wrong: %v", metadata)
	}
}
