package extproc

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
)

// Semantic capabilities stay required even when scenario IDs or counts change.
var requiredProtectionCoverage = []string{
	"active-tool-call",
	"tool-result",
	"tool-followup",
	"tool-release",
	"provider-bound",
	"provider-release",
	"small-advantage",
	"clear-advantage",
	"warm-cache",
	"candidate-boundary",
	"session-continuity",
	"conversation-isolation",
	"first-request",
	"missing-identity",
	"observe",
	"bypass",
}

func validateProtectionCoverage(scenarios []protectionScenario) error {
	seen := map[string]bool{}
	for _, scenario := range scenarios {
		for _, step := range scenario.Steps {
			for _, tag := range step.Coverage {
				if !slices.Contains(requiredProtectionCoverage, tag) {
					return fmt.Errorf("unknown protection coverage %q", tag)
				}
				seen[tag] = true
			}
		}
	}
	for _, tag := range requiredProtectionCoverage {
		if !seen[tag] {
			return fmt.Errorf("missing protection coverage %q", tag)
		}
	}
	return nil
}

func TestRouterLearningSessionRequiredCoverageCannotDisappear(t *testing.T) {
	for _, removed := range requiredProtectionCoverage {
		t.Run(removed, func(t *testing.T) {
			corpus, _ := loadProtectionCorpus(t)
			for i := range corpus.Scenarios {
				for j := range corpus.Scenarios[i].Steps {
					step := &corpus.Scenarios[i].Steps[j]
					step.Coverage = slices.DeleteFunc(step.Coverage, func(tag string) bool { return tag == removed })
				}
			}
			if err := validateProtectionScenarios(corpus.Scenarios); err == nil {
				t.Fatal("missing required capability accepted")
			}
		})
	}
}

func TestRouterLearningSessionDiagnosticRegressionsFail(t *testing.T) {
	expected := protectionExpectation{Model: "a", Sampling: false, Action: "hold_current", Reason: "tool_or_protocol_state", PreflightReason: "tool_or_protocol_state", HardLocked: extprocBoolPtr(true)}
	row := protectionRow{Selected: "a", Action: expected.Action, Reason: expected.Reason, PreflightReason: expected.PreflightReason, HardLocked: true}
	if failures := protectionFailures(row, expected); len(failures) != 0 {
		t.Fatal(failures)
	}
	row.HardLocked = false
	if len(protectionFailures(row, expected)) != 1 {
		t.Fatal("hard-lock regression escaped")
	}
	row.HardLocked = true
	row.PreflightReason = "wrong"
	if len(protectionFailures(row, expected)) != 1 {
		t.Fatal("preflight regression escaped")
	}
}

func TestRouterLearningSessionYAMLRejectsMalformedDocuments(t *testing.T) {
	raw, err := os.ReadFile(protectionCorpusPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []string{
		string(raw) + "unknown: true\n",
		string(raw) + "schema_version: duplicate\n",
		string(raw) + "---\n{}\n",
		strings.Replace(string(raw), "          hard_locked: false\n", "", 1),
		strings.Replace(string(raw), "          preflight_reason: no_tool_or_protocol_state\n", "", 1),
	} {
		if _, err := decodeProtectionCorpus([]byte(invalid)); err == nil {
			t.Fatal("malformed or incomplete YAML accepted")
		}
	}
}
