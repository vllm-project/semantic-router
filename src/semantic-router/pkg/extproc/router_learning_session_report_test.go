package extproc

import (
	"fmt"
	"slices"
)

type protectionRow struct {
	Scenario        string   `json:"scenario"`
	Step            string   `json:"step"`
	Turn            int      `json:"turn"`
	Category        string   `json:"category"`
	Previous        string   `json:"previous_model"`
	Proposal        string   `json:"proposed_model"`
	Selected        string   `json:"final_model"`
	CandidateCount  int      `json:"candidate_count"`
	SamplingAllowed bool     `json:"sampling_allowed"`
	PreflightReason string   `json:"preflight_reason"`
	Action          string   `json:"replay_action"`
	Reason          string   `json:"replay_reason"`
	HardLocked      bool     `json:"hard_locked"`
	CacheWarmth     float64  `json:"cache_warmth_input"`
	Failures        []string `json:"failures"`
}

type protectionRate struct {
	Count int      `json:"count"`
	Total int      `json:"total"`
	Rate  *float64 `json:"rate"`
}

type protectionReport struct {
	Schema          string                    `json:"schema_version"`
	CorpusSHA256    string                    `json:"corpus_sha256"`
	Evidence        string                    `json:"evidence"`
	Passed          bool                      `json:"passed"`
	Deterministic   bool                      `json:"deterministic"`
	Scenarios       int                       `json:"scenarios"`
	Turns           int                       `json:"turns"`
	Metrics         map[string]protectionRate `json:"metrics"`
	Unavailable     map[string]string         `json:"unavailable_metrics"`
	MissingCoverage []string                  `json:"missing_coverage"`
	Rows            []protectionRow           `json:"rows"`
}

func protectionFailures(row protectionRow, expected protectionExpectation) []string {
	failures := []string{}
	if slices.Contains([]string{"blocked", "hold", "opportunity", "boundary"}, expected.Category) && row.Previous == "" {
		failures = append(failures, "continuation scenario did not retain the previous model")
	}
	for _, check := range []struct{ field, got, want string }{
		{"model", row.Selected, expected.Model},
		{"action", row.Action, expected.Action},
		{"reason", row.Reason, expected.Reason},
		{"preflight_reason", row.PreflightReason, expected.PreflightReason},
	} {
		if check.got != check.want {
			failures = append(failures, fmt.Sprintf("%s: got %q, want %q", check.field, check.got, check.want))
		}
	}
	if row.SamplingAllowed != expected.Sampling {
		failures = append(failures, "sampling permission differs from expectation")
	}
	if expected.HardLocked == nil || row.HardLocked != *expected.HardLocked {
		failures = append(failures, "hard-lock status differs from expectation")
	}
	return failures
}

func summarizeProtection(corpus protectionCorpus, digest string, rows []protectionRow) protectionReport {
	report := protectionReport{
		Schema: "agent-routing-protection-report.v1", CorpusSHA256: digest,
		Evidence: "production-protection/scripted-proposals/no-model-execution",
		Passed:   len(rows) > 0, Scenarios: len(corpus.Scenarios), Turns: len(rows), Rows: rows,
		MissingCoverage: corpus.MissingCoverage,
		Metrics:         map[string]protectionRate{},
		Unavailable: map[string]string{
			"quality_benefit": "No model execution or paired task outcomes.",
			"cost_delta":      "No provider billing measurements.",
			"latency_delta":   "Wall-clock test duration is not inference latency.",
			"cache_impact":    "Cache warmth is scripted state, not measured cache savings.",
			"uncertainty":     "Fixed contract scenarios are not a sampled population of agent tasks.",
		},
	}
	for _, key := range []string{"contract_pass", "switch", "blocked_switch_violation", "unsafe_sampling_violation", "missed_scripted_opportunity", "unnecessary_switch", "replay_explainability"} {
		report.Metrics[key] = protectionRate{}
	}
	for _, row := range rows {
		passed := len(row.Failures) == 0
		report.Passed = report.Passed && passed
		report.add("contract_pass", passed)
		report.add("replay_explainability", row.Action != "" && row.Reason != "")
		switched := row.Previous != "" && row.Previous != row.Selected
		if row.Previous != "" {
			report.add("switch", switched)
		}
		if row.Category == "blocked" {
			report.add("blocked_switch_violation", switched)
			report.add("unsafe_sampling_violation", row.SamplingAllowed)
		}
		if row.Category == "opportunity" {
			report.add("missed_scripted_opportunity", row.Selected != row.Proposal)
		}
		if slices.Contains([]string{"blocked", "hold"}, row.Category) {
			report.add("unnecessary_switch", switched)
		}
	}
	return report
}

func (report *protectionReport) add(key string, positive bool) {
	metric := report.Metrics[key]
	metric.Total++
	if positive {
		metric.Count++
	}
	rate := float64(metric.Count) / float64(metric.Total)
	metric.Rate = &rate
	report.Metrics[key] = metric
}
