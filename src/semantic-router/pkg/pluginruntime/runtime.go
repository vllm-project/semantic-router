package pluginruntime

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type Binding struct {
	Recipe   config.RecipeName `json:"recipe"`
	Decision string            `json:"decision"`
}

type ExecutionMode string

const (
	ModePreview ExecutionMode = "preview"
	ModeProbe   ExecutionMode = "probe"
)

var (
	ErrUnavailable    = errors.New("plugin runtime dependency is unavailable")
	ErrInvalidBinding = errors.New("plugin binding does not exist in the specified recipe and decision")
	ErrProbeRequired  = errors.New("this operation requires mode=probe to invoke configured dependencies")
)

// Guarantees describes one operation's actual effects. A backend call means
// invoking a configured inference/retrieval backend, which may run remotely.
// Neither mode may generate a provider response or write plugin storage.
type Guarantees struct {
	Mode         ExecutionMode `json:"mode"`
	Persisted    bool          `json:"persisted"`
	BackendCalls bool          `json:"backend_calls"`
}

type Capabilities struct {
	Inspector BindingInspector
	Guards    GuardPreviewRuntime
	Retrieval RetrievalPreviewRuntime
}

type GuardPreviewRuntime interface {
	PreviewResponseGuard(context.Context, ResponseGuardPreviewRequest) (ResponseGuardPreviewResponse, error)
}

// ResponseGuardPreviewRequest is an answer for a binding's response_guard to
// read: the question it answered, its grounding context (retrieved documents
// and tool results) and the answer itself.
type ResponseGuardPreviewRequest struct {
	Binding  Binding       `json:"binding"`
	Mode     ExecutionMode `json:"mode,omitempty"`
	Question string        `json:"question"`
	Context  string        `json:"context,omitempty"`
	Response string        `json:"response"`
}

// ResponseGuardPreviewResponse is what the guard does with the answer: block,
// annotate with warnings, or none, each rule's outcome in order up to the
// first that blocks, and the response-stage signals the rules read.
type ResponseGuardPreviewResponse struct {
	Guarantees
	Binding  Binding                      `json:"binding"`
	Action   string                       `json:"action"`
	Warnings []string                     `json:"warnings,omitempty"`
	Reason   string                       `json:"reason,omitempty"`
	Rules    []ResponseGuardRuleOutcome   `json:"rules,omitempty"`
	Signals  []ResponseGuardSignalOutcome `json:"signals,omitempty"`
}

// ResponseGuardRuleOutcome is one rule's reading of the answer: matched,
// not_matched or unknown, the action it applies and the warning codes it
// annotates with.
type ResponseGuardRuleOutcome struct {
	State        string   `json:"state"`
	Action       string   `json:"action,omitempty"`
	Warnings     []string `json:"warnings,omitempty"`
	MatchedRules []string `json:"matched_rules,omitempty"`
}

// ResponseGuardSignalOutcome is one response-stage rule's reading of the
// answer, under its type:name key: detected, not_detected or unavailable
// (with the failure code as reason), its score, and the evidence behind it.
type ResponseGuardSignalOutcome struct {
	Signal           string   `json:"signal"`
	Verdict          string   `json:"verdict"`
	Reason           string   `json:"reason,omitempty"`
	Score            *float64 `json:"score,omitempty"`
	UnsupportedSpans []string `json:"unsupported_spans,omitempty"`
}

func NormalizeMode(mode ExecutionMode) (ExecutionMode, error) {
	switch mode {
	case "", ModePreview:
		return ModePreview, nil
	case ModeProbe:
		return ModeProbe, nil
	default:
		return "", errors.New("mode must be preview or probe")
	}
}
