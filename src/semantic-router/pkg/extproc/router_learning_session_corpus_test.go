package extproc

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"slices"
	"testing"

	"gopkg.in/yaml.v3"
)

const protectionCorpusPath = "testdata/router_learning_sessions.v1.yaml"

type protectionCorpus struct {
	Schema          string               `json:"schema_version" yaml:"schema_version"`
	MissingCoverage []string             `json:"missing_coverage" yaml:"missing_coverage"`
	Scenarios       []protectionScenario `json:"scenarios" yaml:"scenarios"`
}

type protectionScenario struct {
	ID    string           `json:"id" yaml:"id"`
	Scope string           `json:"scope" yaml:"scope"`
	Mode  string           `json:"mode" yaml:"mode"`
	Steps []protectionStep `json:"steps" yaml:"steps"`
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
	Expected           protectionExpectation `json:"expected" yaml:"expected"`
}

type protectionMessage struct {
	Role       string `json:"role" yaml:"role"`
	Text       string `json:"text" yaml:"text"`
	ToolCallID string `json:"tool_call_id" yaml:"tool_call_id"`
}

type protectionExpectation struct {
	HardLocked      *bool  `json:"hard_locked" yaml:"hard_locked"`
	PreflightReason string `json:"preflight_reason" yaml:"preflight_reason"`
	Model           string `json:"model" yaml:"model"`
	Sampling        bool   `json:"sampling_allowed" yaml:"sampling_allowed"`
	Action          string `json:"action" yaml:"action"`
	Reason          string `json:"reason" yaml:"reason"`
	Category        string `json:"category" yaml:"category"`
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
	if err := decoder.Decode(new(any)); err != io.EOF {
		return corpus, fmt.Errorf("corpus must contain exactly one YAML document")
	}
	if corpus.Schema != "agent-routing-protection.v1" || len(corpus.Scenarios) == 0 || len(corpus.MissingCoverage) == 0 {
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
		stepIDs := map[string]bool{}
		for _, step := range scenario.Steps {
			if err := validateProtectionStep(step, stepIDs); err != nil {
				return fmt.Errorf("%s: %w", scenario.ID, err)
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
	if !slices.Contains(step.Candidates, step.Proposal) || !slices.Contains(step.Candidates, step.Expected.Model) {
		return fmt.Errorf("%s: proposal and expectation must be eligible", step.ID)
	}
	if step.Expected.Action == "" || step.Expected.Reason == "" || step.Expected.PreflightReason == "" || step.Expected.HardLocked == nil {
		return fmt.Errorf("%s: missing assertion", step.ID)
	}
	if !slices.Contains([]string{"baseline", "blocked", "opportunity", "hold", "boundary", "observe", "bypass", "missing_identity"}, step.Expected.Category) {
		return fmt.Errorf("%s: unknown category", step.ID)
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
