package evaluationplane

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCommandProcessRouterLearningEndToEnd(t *testing.T) {
	report, records := runCommandProcessReplay(t, CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "production routing sampling replay",
		SuiteIDs: []string{"router-learning-core"}, TrackIDs: []TrackID{"joint"},
		Mode: ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 12, Concurrency: 1, Seed: 17,
	})
	if report.Provenance.BenchmarkRevisions["router-learning-core"] != "router-learning-core-v2" {
		t.Fatal("report lost its versioned production replay corpus identity")
	}
	metrics := map[string]Metric{}
	for _, metric := range report.Metrics {
		metrics[metric.ID] = metric
	}
	for _, policy := range routerLearningPolicyIDs {
		prefix := "joint.router_learning." + policy + "."
		solve := metrics[prefix+"solve_rate"]
		if solve.Value == nil || solve.SampleCount != 384 || len(solve.ConfidenceInterval) != 2 {
			t.Fatalf("missing seeded solve-rate evidence for %s: %+v", policy, solve)
		}
		for _, name := range []string{"protection_violation_rate", "hard_constraint_violation_rate", "propensity_coverage"} {
			metric := metrics[prefix+name]
			if metric.Value == nil || *metric.Value != 0 {
				t.Fatalf("unexpected %s: %+v", name, metric)
			}
		}
	}
	rows := bytes.Split(bytes.TrimSpace(records), []byte("\n"))
	if len(rows) != 1152 {
		t.Fatalf("want 1152 paired method rows, got %d", len(rows))
	}
	counts := map[string]int{}
	for _, row := range rows {
		var record executionRecordEvidence
		if err := json.Unmarshal(row, &record); err != nil {
			t.Fatal(err)
		}
		if record.RouterLearning == nil {
			t.Fatal("missing learning method evidence")
		}
		counts[record.RouterLearning.PolicyID]++
	}
	for _, policy := range routerLearningPolicyIDs {
		if counts[policy] != 384 {
			t.Fatalf("unpaired policy %s: %d", policy, counts[policy])
		}
	}
	if strings.Contains(string(records), "simplified-routing-sampling") {
		t.Fatal("production replay published the superseded simplified policy")
	}
}
