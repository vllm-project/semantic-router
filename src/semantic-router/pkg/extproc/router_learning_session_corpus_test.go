package extproc

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const protectionCorpusPath = "testdata/router_learning_sessions.v2.yaml"

type protectionCorpus struct {
	Schema          string               `json:"schema_version" yaml:"schema_version"`
	MissingCoverage []string             `json:"missing_coverage" yaml:"missing_coverage"`
	Scenarios       []protectionScenario `json:"scenarios" yaml:"scenarios"`
}

type protectionScenario struct {
	ID           string                  `json:"id" yaml:"id"`
	Scope        string                  `json:"scope" yaml:"scope"`
	Mode         string                  `json:"mode" yaml:"mode"`
	ProgressGate *protectionProgressGate `json:"progress_gate,omitempty" yaml:"progress_gate,omitempty"`
	Steps        []protectionStep        `json:"steps" yaml:"steps"`
}

// protectionProgressGate keeps every benchmark knob explicit. Zero is a valid
// policy value, so the corpus does not inherit ambient product defaults.
type protectionProgressGate struct {
	Mode                      string  `json:"mode" yaml:"mode"`
	CalibrationID             string  `json:"calibration_id" yaml:"calibration_id"`
	WindowSize                int     `json:"window_size" yaml:"window_size"`
	WindowTTLSeconds          int     `json:"window_ttl_seconds" yaml:"window_ttl_seconds"`
	MinWindowOutcomes         int     `json:"min_window_outcomes" yaml:"min_window_outcomes"`
	MinConsecutiveRegressions int     `json:"min_consecutive_regressions" yaml:"min_consecutive_regressions"`
	MinConsecutiveRecoveries  int     `json:"min_consecutive_recoveries" yaml:"min_consecutive_recoveries"`
	CooldownSeconds           float64 `json:"cooldown_seconds" yaml:"cooldown_seconds"`
	MaxSwitchesPerWindow      int     `json:"max_switches_per_window" yaml:"max_switches_per_window"`
}

type protectionStep struct {
	Coverage           []string              `json:"coverage" yaml:"coverage"`
	ID                 string                `json:"id" yaml:"id"`
	Messages           []protectionMessage   `json:"append_messages" yaml:"append_messages"`
	Candidates         []string              `json:"candidates" yaml:"candidates"`
	Proposal           string                `json:"proposal" yaml:"proposal"`
	Scores             map[string]float64    `json:"scores" yaml:"scores"`
	PreviousResponseID string                `json:"previous_response_id" yaml:"previous_response_id"`
	CacheWarmth        float64               `json:"cache_warmth" yaml:"cache_warmth"`
	MissingIdentity    bool                  `json:"missing_identity" yaml:"missing_identity"`
	Conversation       string                `json:"conversation" yaml:"conversation"`
	Outcome            string                `json:"outcome,omitempty" yaml:"outcome,omitempty"`
	Expected           protectionExpectation `json:"expected" yaml:"expected"`
}

type protectionMessage struct {
	Role       string `json:"role" yaml:"role"`
	Text       string `json:"text" yaml:"text"`
	ToolCallID string `json:"tool_call_id" yaml:"tool_call_id"`
}

type protectionExpectation struct {
	Rejected        bool                       `json:"rejected" yaml:"rejected"`
	HardLocked      *bool                      `json:"hard_locked" yaml:"hard_locked"`
	PreflightReason string                     `json:"preflight_reason" yaml:"preflight_reason"`
	Model           string                     `json:"model" yaml:"model"`
	Sampling        bool                       `json:"sampling_allowed" yaml:"sampling_allowed"`
	Action          string                     `json:"action" yaml:"action"`
	Reason          string                     `json:"reason" yaml:"reason"`
	Category        string                     `json:"category" yaml:"category"`
	Gate            *protectionGateExpectation `json:"gate,omitempty" yaml:"gate,omitempty"`
}

type protectionGateExpectation struct {
	Decision          string `json:"decision" yaml:"decision"`
	Reason            string `json:"reason" yaml:"reason"`
	Origin            string `json:"origin" yaml:"origin"`
	CalibrationID     string `json:"calibration_id" yaml:"calibration_id"`
	ApplicationReason string `json:"application_reason" yaml:"application_reason"`
	Applied           bool   `json:"applied" yaml:"applied"`
	Enforced          bool   `json:"enforced" yaml:"enforced"`
	ColdStart         bool   `json:"cold_start" yaml:"cold_start"`
	AttributableCount int    `json:"attributable_count" yaml:"attributable_count"`
	MissingCount      int    `json:"missing_count" yaml:"missing_count"`
	WindowCount       int    `json:"window_count" yaml:"window_count"`
	RegressionStreak  int    `json:"regression_streak" yaml:"regression_streak"`
	RecoveryStreak    int    `json:"recovery_streak" yaml:"recovery_streak"`
	SwitchesInWindow  int    `json:"switches_in_window" yaml:"switches_in_window"`
}

func loadProtectionCorpus(t *testing.T) (protectionCorpus, string) {
	t.Helper()
	raw, err := os.ReadFile(protectionCorpusPath)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := decodeProtectionCorpus(raw)
	if err != nil {
		t.Fatal(err)
	}
	return corpus, fmt.Sprintf("%x", sha256.Sum256(raw))
}

func decodeProtectionCorpus(raw []byte) (protectionCorpus, error) {
	var corpus protectionCorpus
	decoder := yaml.NewDecoder(bytes.NewReader(raw))
	decoder.KnownFields(true)
	if err := decoder.Decode(&corpus); err != nil {
		return corpus, err
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return corpus, fmt.Errorf("corpus must contain exactly one YAML document")
	}
	if corpus.Schema != "agent-routing-protection.v2" || len(corpus.Scenarios) == 0 || len(corpus.MissingCoverage) == 0 {
		return corpus, fmt.Errorf("corpus requires a supported version, scenarios and missing coverage")
	}
	return corpus, validateProtectionScenarios(corpus.Scenarios)
}

func validateProtectionScenarios(scenarios []protectionScenario) error {
	ids := map[string]bool{}
	categories := map[string]bool{}
	for _, scenario := range scenarios {
		if scenario.ID == "" || ids[scenario.ID] || len(scenario.Steps) == 0 {
			return fmt.Errorf("invalid or repeated scenario %q", scenario.ID)
		}
		ids[scenario.ID] = true
		if !slices.Contains([]string{"conversation", "session"}, scenario.Scope) || !slices.Contains([]string{"apply", "observe", "bypass"}, scenario.Mode) {
			return fmt.Errorf("invalid scope/mode in %s", scenario.ID)
		}
		if err := validateProtectionProgressGate(scenario.ProgressGate); err != nil {
			return fmt.Errorf("%s: %w", scenario.ID, err)
		}
		stepIDs := map[string]bool{}
		for _, step := range scenario.Steps {
			if err := validateProtectionStep(step, stepIDs); err != nil {
				return fmt.Errorf("%s: %w", scenario.ID, err)
			}
			if step.Expected.Gate != nil && scenario.ProgressGate == nil {
				return fmt.Errorf("%s/%s: gate expectation requires gate config", scenario.ID, step.ID)
			}
			categories[step.Expected.Category] = true
		}
	}
	for _, category := range []string{"baseline", "blocked", "opportunity", "hold", "boundary", "observe", "bypass", "missing_identity"} {
		if !categories[category] {
			return fmt.Errorf("missing required category %s", category)
		}
	}
	return validateProtectionCoverage(scenarios)
}

func validateProtectionProgressGate(gate *protectionProgressGate) error {
	if gate == nil {
		return nil
	}
	name, version, versioned := strings.Cut(gate.CalibrationID, "@")
	if !slices.Contains([]string{"observe", "enforce"}, gate.Mode) || !versioned || name == "" || version == "" ||
		strings.ContainsAny(gate.CalibrationID, " \t\n\r") || gate.WindowSize < 1 || gate.WindowSize > 256 ||
		gate.WindowTTLSeconds < 1 || gate.WindowTTLSeconds > 86400 || gate.MinWindowOutcomes < 0 ||
		gate.MinConsecutiveRegressions < 0 || gate.MinConsecutiveRecoveries < 0 ||
		gate.MinWindowOutcomes > gate.WindowSize || gate.MinConsecutiveRegressions > gate.WindowSize ||
		gate.MinConsecutiveRecoveries > gate.WindowSize || gate.MinConsecutiveRegressions < gate.MinConsecutiveRecoveries ||
		gate.CooldownSeconds < 0 || gate.MaxSwitchesPerWindow < 0 || gate.MaxSwitchesPerWindow > 256 {
		return fmt.Errorf("invalid progress gate")
	}
	return nil
}

func validateProtectionStep(step protectionStep, ids map[string]bool) error {
	if step.ID == "" || ids[step.ID] || step.Conversation == "" || len(step.Messages) == 0 || len(step.Candidates) == 0 {
		return fmt.Errorf("invalid or repeated step %q", step.ID)
	}
	ids[step.ID] = true
	if step.CacheWarmth < 0 || step.CacheWarmth > 1 {
		return fmt.Errorf("%s: invalid warmth", step.ID)
	}
	if err := validateProtectionCandidates(step); err != nil {
		return err
	}
	if err := validateProtectionExpectation(step); err != nil {
		return err
	}
	return validateProtectionMessages(step)
}

func validateProtectionCandidates(step protectionStep) error {
	seen := map[string]bool{}
	for _, model := range step.Candidates {
		score, ok := step.Scores[model]
		if !ok || score < 0 || score > 1 || seen[model] || !slices.Contains([]string{"protection-cheap", "protection-frontier"}, model) {
			return fmt.Errorf("%s: invalid candidate/score", step.ID)
		}
		seen[model] = true
	}
	if len(step.Scores) != len(seen) {
		return fmt.Errorf("%s: scores escape candidates", step.ID)
	}
	return nil
}

func validateProtectionExpectation(step protectionStep) error {
	if !slices.Contains(step.Candidates, step.Proposal) ||
		(!step.Expected.Rejected && !slices.Contains(step.Candidates, step.Expected.Model)) ||
		(step.Expected.Rejected && step.Expected.Model != "") {
		return fmt.Errorf("%s: proposal and expectation must be eligible", step.ID)
	}
	if step.Expected.Action == "" || step.Expected.Reason == "" || step.Expected.PreflightReason == "" || step.Expected.HardLocked == nil {
		return fmt.Errorf("%s: missing assertion", step.ID)
	}
	if !slices.Contains([]string{"baseline", "blocked", "opportunity", "hold", "boundary", "observe", "bypass", "missing_identity"}, step.Expected.Category) {
		return fmt.Errorf("%s: unknown category", step.ID)
	}
	if step.Outcome != "" && !slices.Contains([]string{"progress", "no_progress", "regression", "provider_error", "tool_error", "missing"}, step.Outcome) {
		return fmt.Errorf("%s: unknown outcome", step.ID)
	}
	if gate := step.Expected.Gate; gate != nil {
		if !slices.Contains([]string{"switch", "suppress"}, gate.Decision) ||
			!slices.Contains([]string{"escalation", "downgrade"}, gate.Origin) ||
			gate.CalibrationID == "" || gate.ApplicationReason == "" {
			return fmt.Errorf("%s: incomplete gate expectation", step.ID)
		}
	}
	return nil
}

func validateProtectionMessages(step protectionStep) error {
	for _, message := range step.Messages {
		if !slices.Contains([]string{"user", "assistant", "tool"}, message.Role) || message.Text == "" {
			return fmt.Errorf("%s: invalid message", step.ID)
		}
		if (message.Role == "tool" && message.ToolCallID == "") || (message.Role == "user" && message.ToolCallID != "") {
			return fmt.Errorf("%s: invalid tool exchange", step.ID)
		}
	}
	return nil
}
