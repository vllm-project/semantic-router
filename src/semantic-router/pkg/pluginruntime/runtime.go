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
	PreviewResponseJailbreak(context.Context, ResponseJailbreakPreviewRequest) (GuardPreviewResponse, error)
	PreviewHallucination(context.Context, HallucinationPreviewRequest) (GuardPreviewResponse, error)
}

type ResponseJailbreakPreviewRequest struct {
	Binding  Binding       `json:"binding"`
	Mode     ExecutionMode `json:"mode,omitempty"`
	Response string        `json:"response"`
}

type HallucinationPreviewRequest struct {
	Binding         Binding       `json:"binding"`
	Mode            ExecutionMode `json:"mode,omitempty"`
	Question        string        `json:"question"`
	Context         string        `json:"context"`
	Response        string        `json:"response"`
	FactCheckNeeded bool          `json:"fact_check_needed"`
}

type GuardPreviewResponse struct {
	Guarantees
	Binding          Binding  `json:"binding"`
	Enabled          bool     `json:"enabled"`
	Eligible         bool     `json:"eligible"`
	Resolved         bool     `json:"resolved"`
	Detected         bool     `json:"detected"`
	Action           string   `json:"action"`
	Reason           string   `json:"reason,omitempty"`
	DetectionSource  string   `json:"detection_source"`
	Score            *float32 `json:"score,omitempty"`
	ScoreKind        string   `json:"score_kind,omitempty"`
	Label            string   `json:"label,omitempty"`
	MatchedRules     []string `json:"matched_rules,omitempty"`
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
