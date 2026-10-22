package extproc

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type protectionRow struct {
	Rejected        bool               `json:"rejected"`
	Scenario        string             `json:"scenario"`
	Step            string             `json:"step"`
	Turn            int                `json:"turn"`
	Category        string             `json:"category"`
	Previous        string             `json:"previous_model"`
	Proposal        string             `json:"proposed_model"`
	Selected        string             `json:"final_model"`
	CandidateCount  int                `json:"candidate_count"`
	SamplingAllowed bool               `json:"sampling_allowed"`
	PreflightReason string             `json:"preflight_reason"`
	Action          string             `json:"replay_action"`
	Reason          string             `json:"replay_reason"`
	HardLocked      bool               `json:"hard_locked"`
	CacheWarmth     float64            `json:"cache_warmth_input"`
	Gate            *protectionGateRow `json:"progress_gate,omitempty"`
	Failures        []string           `json:"failures"`
}

type protectionGateRow struct {
	Decision          string `json:"decision"`
	Reason            string `json:"reason"`
	Origin            string `json:"origin"`
	Mode              string `json:"mode"`
	CalibrationID     string `json:"calibration_id"`
	EvidenceVersion   string `json:"evidence_version"`
	ApplicationReason string `json:"application_reason"`
	Applied           bool   `json:"applied"`
	Enforced          bool   `json:"enforced"`
	ColdStart         bool   `json:"cold_start"`
	AttributableCount int    `json:"attributable_count"`
	MissingCount      int    `json:"missing_count"`
	WindowCount       int    `json:"window_count"`
	RegressionStreak  int    `json:"regression_streak"`
	RecoveryStreak    int    `json:"recovery_streak"`
	SwitchesInWindow  int    `json:"switches_in_window"`
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
	if row.Rejected != expected.Rejected {
		failures = append(failures, "rejection outcome differs from expectation")
	}
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
	if expected.Gate == nil && row.Gate != nil {
		failures = append(failures, "unexpected progress-gate verdict")
	}
	if expected.Gate != nil {
		if row.Gate == nil {
			return append(failures, "missing progress-gate verdict")
		}
		for _, check := range []struct{ field, got, want string }{
			{"gate decision", row.Gate.Decision, expected.Gate.Decision},
			{"gate reason", row.Gate.Reason, expected.Gate.Reason},
			{"gate origin", row.Gate.Origin, expected.Gate.Origin},
			{"gate calibration", row.Gate.CalibrationID, expected.Gate.CalibrationID},
			{"gate application", row.Gate.ApplicationReason, expected.Gate.ApplicationReason},
		} {
			if check.got != check.want {
				failures = append(failures, fmt.Sprintf("%s: got %q, want %q", check.field, check.got, check.want))
			}
		}
		wantMode := selection.GateModeObserve
		if expected.Gate.Enforced {
			wantMode = selection.GateModeEnforce
		}
		if row.Gate.Mode != wantMode {
			failures = append(failures, fmt.Sprintf("gate mode: got %q, want %q", row.Gate.Mode, wantMode))
		}
		if row.Gate.EvidenceVersion != selection.ProgressEvidenceVersion {
			failures = append(failures, fmt.Sprintf("gate evidence version: got %q, want %q", row.Gate.EvidenceVersion, selection.ProgressEvidenceVersion))
		}
		for _, check := range []struct {
			field     string
			got, want bool
		}{
			{"gate applied", row.Gate.Applied, expected.Gate.Applied},
			{"gate enforced", row.Gate.Enforced, expected.Gate.Enforced},
			{"gate cold start", row.Gate.ColdStart, expected.Gate.ColdStart},
		} {
			if check.got != check.want {
				failures = append(failures, fmt.Sprintf("%s differs from expectation", check.field))
			}
		}
		for _, check := range []struct {
			field     string
			got, want int
		}{
			{"gate attributable count", row.Gate.AttributableCount, expected.Gate.AttributableCount},
			{"gate missing count", row.Gate.MissingCount, expected.Gate.MissingCount},
			{"gate window count", row.Gate.WindowCount, expected.Gate.WindowCount},
			{"gate regression streak", row.Gate.RegressionStreak, expected.Gate.RegressionStreak},
			{"gate recovery streak", row.Gate.RecoveryStreak, expected.Gate.RecoveryStreak},
			{"gate switches in window", row.Gate.SwitchesInWindow, expected.Gate.SwitchesInWindow},
		} {
			if check.got != check.want {
				failures = append(failures, fmt.Sprintf("%s: got %d, want %d", check.field, check.got, check.want))
			}
		}
	}
	return failures
}

func summarizeProtection(corpus protectionCorpus, digest string, rows []protectionRow) protectionReport {
	report := protectionReport{
		Schema: "agent-routing-protection-report.v2", CorpusSHA256: digest,
		Evidence: "production-protection-and-progress-gate/scripted-proposals-and-outcomes/no-model-execution",
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
	for _, key := range []string{"contract_pass", "switch", "blocked_switch_violation", "unsafe_sampling_violation", "missed_scripted_opportunity", "unnecessary_switch", "replay_explainability", "progress_gate_contract_pass", "progress_gate_explainability", "progress_gate_suppression_applied", "progress_gate_ineligible_hold_avoided"} {
		report.Metrics[key] = protectionRate{}
	}
	for _, row := range rows {
		passed := len(row.Failures) == 0
		report.Passed = report.Passed && passed
		report.add("contract_pass", passed)
		report.add("replay_explainability", row.Action != "" && row.Reason != "")
		if row.Gate != nil {
			report.add("progress_gate_contract_pass", !slices.ContainsFunc(row.Failures, func(failure string) bool {
				return strings.Contains(failure, "progress-gate") || strings.HasPrefix(failure, "gate ")
			}))
			report.add("progress_gate_explainability", row.Gate.Decision != "" && row.Gate.Origin != "" && row.Gate.CalibrationID != "" && row.Gate.ApplicationReason != "")
			if row.Gate.Decision == "suppress" && row.Gate.Enforced && row.Gate.ApplicationReason == "switch_suppressed" {
				report.add("progress_gate_suppression_applied", row.Gate.Applied)
			}
			if strings.HasPrefix(row.Gate.ApplicationReason, "current_ineligible:") {
				report.add("progress_gate_ineligible_hold_avoided", !row.Gate.Applied)
			}
		}
		switched := !row.Rejected && row.Previous != "" && row.Previous != row.Selected
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
