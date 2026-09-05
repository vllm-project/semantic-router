package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// This is a correctness benchmark, not a Go allocation/time benchmark. It runs
// in the normal core test gate, and fails on any per-turn contract mismatch.
func TestRouterLearningSessionBenchmark(t *testing.T) {
	corpus, digest := loadProtectionCorpus(t)
	run := func() protectionReport {
		rows := []protectionRow{}
		for _, scenario := range corpus.Scenarios {
			rows = append(rows, runProtectionScenario(t, scenario)...)
		}
		return summarizeProtection(corpus, digest, rows)
	}
	first, second := run(), run()
	raw, err := json.MarshalIndent(first, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := json.MarshalIndent(second, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	first.Deterministic = bytes.Equal(raw, repeated)
	first.Passed = first.Passed && first.Deterministic
	if !first.Deterministic {
		t.Error("identical corpus produced different reports")
	}
	writeProtectionReport(t, first)
	for _, row := range first.Rows {
		for _, failure := range row.Failures {
			t.Errorf("%s/%s: %s", row.Scenario, row.Step, failure)
		}
	}
	if !first.Passed {
		t.Error("protection benchmark failed; inspect the per-turn report")
	}
	t.Logf("%d scenarios, %d turns; deterministic report; passed=%v", first.Scenarios, first.Turns, first.Passed)
}

func writeProtectionReport(t *testing.T, first protectionReport) {
	t.Helper()
	raw, err := json.MarshalIndent(first, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if output := os.Getenv("ROUTER_PROTECTION_REPORT"); output != "" {
		if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(output, append(raw, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRouterLearningSessionCorpusRejectsInvalidInput(t *testing.T) {
	corpus, _ := loadProtectionCorpus(t)
	for _, mutate := range []struct {
		name  string
		apply func(*protectionCorpus)
	}{
		{"unknown schema", func(c *protectionCorpus) { c.Schema = "future" }},
		{"empty scenarios", func(c *protectionCorpus) { c.Scenarios = nil }},
		{"missing coverage omitted", func(c *protectionCorpus) { c.MissingCoverage = nil }},
		{"duplicate scenario", func(c *protectionCorpus) { c.Scenarios = append(c.Scenarios, c.Scenarios[0]) }},
		{"ineligible expectation", func(c *protectionCorpus) { c.Scenarios[0].Steps[0].Expected.Model = "unconfigured" }},
		{"unknown category", func(c *protectionCorpus) { c.Scenarios[0].Steps[0].Expected.Category = "typo" }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			raw, err := json.Marshal(corpus)
			if err != nil {
				t.Fatal(err)
			}
			var changed protectionCorpus
			if err := json.Unmarshal(raw, &changed); err != nil {
				t.Fatal(err)
			}
			mutate.apply(&changed)
			raw, err = json.Marshal(changed)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeProtectionCorpus(raw); err == nil {
				t.Fatal("invalid corpus accepted")
			}
		})
	}
}

func TestRouterLearningSessionDetectsDisabledProtection(t *testing.T) {
	corpus, digest := loadProtectionCorpus(t)
	for _, scenario := range corpus.Scenarios {
		if scenario.ID != "tool-loop-and-release" {
			continue
		}
		// Negative control: keep a shared session, but disable enforcement. The
		// gate must detect an actual switch during an unfinished tool exchange.
		scenario.Scope = "session"
		scenario.Mode = "bypass"
		report := summarizeProtection(corpus, digest, runProtectionScenario(t, scenario))
		if report.Passed || report.Metrics["blocked_switch_violation"].Count == 0 {
			t.Fatal("benchmark did not detect disabled protection")
		}
		return
	}
	t.Fatal("missing maintained tool-loop negative control")
}

func TestRouterLearningSessionReportDetectsRegressions(t *testing.T) {
	corpus, digest := loadProtectionCorpus(t)
	expected := protectionExpectation{HardLocked: extprocBoolPtr(false), Model: "protection-cheap", Sampling: false, Action: "hold_current", Reason: "tool_or_protocol_state"}
	unsafe := protectionRow{Category: "blocked", Previous: "protection-cheap", Proposal: "protection-frontier", Selected: "protection-frontier", SamplingAllowed: true}
	unsafe.Failures = protectionFailures(unsafe, expected)
	missed := protectionRow{Category: "opportunity", Previous: "protection-cheap", Proposal: "protection-frontier", Selected: "protection-cheap"}
	report := summarizeProtection(corpus, digest, []protectionRow{unsafe, missed})
	if report.Passed || len(unsafe.Failures) != 4 {
		t.Fatal("unsafe route, sampling and missing explanations must fail")
	}
	for _, name := range []string{"blocked_switch_violation", "unsafe_sampling_violation", "missed_scripted_opportunity", "unnecessary_switch"} {
		if metric := report.Metrics[name]; metric.Count != 1 || metric.Total != 1 {
			t.Fatalf("%s: incorrect denominator/count: %+v", name, metric)
		}
	}
	empty := summarizeProtection(corpus, digest, nil)
	if empty.Metrics["missed_scripted_opportunity"].Rate != nil {
		t.Fatal("absent coverage must not report zero violations")
	}
}
