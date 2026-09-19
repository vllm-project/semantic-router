package testcases

import (
	"encoding/json"
	"testing"
)

func TestSrBenchReportRejectsIncompleteOrIncorrectEvidence(t *testing.T) {
	run := dashboardBenchRun{ID: "run-fixture", Status: "completed"}
	run.Manifest.PlanSHA = "frozen-plan"
	for _, scenario := range []struct {
		name   string
		mutate func(map[string]any, map[string]any)
	}{
		{"valid", nil},
		{"different run", func(r, _ map[string]any) { r["run_id"] = "another" }},
		{"incomplete", func(r, _ map[string]any) { r["status"] = "failed" }},
		{"wrong plan", func(r, _ map[string]any) { r["provenance"] = map[string]any{"plan_sha256": "other"} }},
		{"denominator", func(_, s map[string]any) { s["total"] = 1 }},
		{"partial", func(_, s map[string]any) { s["completed"] = 1 }},
		{"reasoning scored", func(_, s map[string]any) { s["correct"] = 0 }},
		{"extra inference", func(_, s map[string]any) { s["request_count"] = 3 }},
		{"unknown cost", func(_, s map[string]any) { s["cost_usd"] = nil }},
		{"incomplete cost", func(_, s map[string]any) { s["cost_complete"] = false }},
		{"incorrect cost", func(_, s map[string]any) { s["cost_usd"] = 0 }},
		{"missing cache write", func(_, s map[string]any) { delete(s["tokens"].(map[string]int), "cache_write_tokens") }},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			score := map[string]any{
				"id": "fixture", "total": 2, "completed": 2, "correct": 2, "accuracy": 1,
				"request_count": 2, "cost_usd": 36.4 / 1_000_000, "cost_complete": true,
				"tokens": map[string]int{"input_tokens": 14, "cached_input_tokens": 4, "cache_write_tokens": 2, "output_tokens": 6},
			}
			report := map[string]any{
				"version": "sr-bench-1.0", "run_id": run.ID, "status": "completed",
				"provenance": map[string]any{"plan_sha256": run.Manifest.PlanSHA},
				"summary":    map[string]any{"targets": []any{score}},
			}
			if scenario.mutate != nil {
				scenario.mutate(report, score)
			}
			raw, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			err = verifySrBenchReport(raw, run)
			if (err != nil) != (scenario.mutate != nil) {
				t.Fatalf("verification error=%v; invalid=%t", err, scenario.mutate != nil)
			}
		})
	}
}
