package extproc

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"testing"
)

const protectionCorpusPath = "testdata/router_learning_sessions.v1.json"

type protectionCorpus struct {
	Schema          string               `json:"schema_version"`
	MissingCoverage []string             `json:"missing_coverage"`
	Scenarios       []protectionScenario `json:"scenarios"`
}

type protectionScenario struct {
	ID    string           `json:"id"`
	Scope string           `json:"scope"`
	Mode  string           `json:"mode"`
	Steps []protectionStep `json:"steps"`
}

type protectionStep struct {
	ID                 string                `json:"id"`
	Messages           []protectionMessage   `json:"append_messages"`
	Candidates         []string              `json:"candidates"`
	Proposal           string                `json:"proposal"`
	Scores             map[string]float64    `json:"scores"`
	PreviousResponseID string                `json:"previous_response_id"`
	CacheWarmth        float64               `json:"cache_warmth"`
	MissingIdentity    bool                  `json:"missing_identity"`
	Conversation       string                `json:"conversation"`
	Expected           protectionExpectation `json:"expected"`
}

type protectionMessage struct {
	Role       string `json:"role"`
	Text       string `json:"text"`
	ToolCallID string `json:"tool_call_id"`
}

type protectionExpectation struct {
	Model    string `json:"model"`
	Sampling bool   `json:"sampling_allowed"`
	Action   string `json:"action"`
	Reason   string `json:"reason"`
	Category string `json:"category"`
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
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&corpus); err != nil {
		return corpus, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return corpus, fmt.Errorf("corpus must contain exactly one JSON object")
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
	if !slices.Contains(step.Candidates, step.Proposal) || !slices.Contains(step.Candidates, step.Expected.Model) {
		return fmt.Errorf("%s: proposal and expectation must be eligible", step.ID)
	}
	if step.Expected.Action == "" || step.Expected.Reason == "" {
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
