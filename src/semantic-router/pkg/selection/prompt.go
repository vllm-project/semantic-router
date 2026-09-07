package selection

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	ErrPromptInvocation          = errors.New("prompt selector invocation failed")
	ErrPromptInvalidOutput       = errors.New("prompt selector output is invalid")
	ErrPromptUndeclaredCandidate = errors.New("prompt selector chose undeclared candidate")
)

// PromptInvoke calls the concrete helper model without running model selection
// recursively. The response must contain only the selector JSON contract.
type PromptInvoke func(
	ctx context.Context,
	model string,
	systemPrompt string,
	input string,
) (PromptInvocationResult, error)

type PromptInvocationResult struct {
	Content          string
	Model            string
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	LatencyMs        int64
}

// PromptSelector uses a small concrete model to select one of the decision's
// candidate ModelRefs. All output protocol details are runtime-owned.
type PromptSelector struct {
	config       config.PromptSelectionConfig
	invoke       PromptInvoke
	descriptions map[string]string
}

func NewPromptSelector(
	cfg config.PromptSelectionConfig,
	invoke PromptInvoke,
	descriptions map[string]string,
) *PromptSelector {
	return &PromptSelector{
		config:       cfg,
		invoke:       invoke,
		descriptions: descriptions,
	}
}

func (s *PromptSelector) Method() SelectionMethod { return MethodPrompt }

func (s *PromptSelector) Tier() AlgorithmTier { return TierExperimental }

func (s *PromptSelector) ExternalDependencies() []Dependency {
	return []Dependency{{
		Name:        "Prompt helper model via looper",
		Type:        DependencyExternalService,
		Description: "Requires a reachable OpenAI-compatible helper model",
		Required:    true,
	}}
}

func (s *PromptSelector) UpdateFeedback(context.Context, *Feedback) error { return nil }

func (s *PromptSelector) Select(
	ctx context.Context,
	selCtx *SelectionContext,
) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}
	if err := validatePromptCandidateModels(selCtx.CandidateModels); err != nil {
		return nil, err
	}
	if s.invoke == nil {
		return nil, fmt.Errorf("prompt selector model invoker is not configured")
	}

	invocation, err := s.invoke(
		ctx,
		s.config.Model,
		s.systemPrompt(selCtx.CandidateModels),
		selCtx.Query,
	)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrPromptInvocation, err)
	}

	choice, err := parsePromptChoice(invocation.Content)
	if err != nil {
		return nil, err
	}
	if candidate := promptCandidateByModel(selCtx.CandidateModels, choice.SelectedModel); candidate != nil {
		return s.selectionResult(
			selCtx.CandidateModels,
			*candidate,
			choice.Rationale,
			invocation,
		), nil
	}
	return nil, ErrPromptUndeclaredCandidate
}

func validatePromptCandidateModels(candidates []config.ModelRef) error {
	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		if _, exists := seen[candidate.Model]; exists {
			return fmt.Errorf(
				"prompt selector requires unique candidate models; duplicate %q",
				candidate.Model,
			)
		}
		seen[candidate.Model] = struct{}{}
	}
	return nil
}

type promptChoice struct {
	SelectedModel string `json:"selected_model"`
	Rationale     string `json:"rationale"`
}

func parsePromptChoice(reply string) (promptChoice, error) {
	content := strings.TrimSpace(reply)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		return promptChoice{}, fmt.Errorf("%w: JSON decode failed", ErrPromptInvalidOutput)
	}
	if len(raw) != 2 || raw["selected_model"] == nil || raw["rationale"] == nil {
		return promptChoice{}, fmt.Errorf("%w: unexpected response fields", ErrPromptInvalidOutput)
	}
	var choice promptChoice
	if err := json.Unmarshal([]byte(content), &choice); err != nil {
		return promptChoice{}, fmt.Errorf("%w: response decode failed", ErrPromptInvalidOutput)
	}
	choice.SelectedModel = strings.TrimSpace(choice.SelectedModel)
	choice.Rationale = strings.TrimSpace(choice.Rationale)
	if choice.SelectedModel == "" || choice.Rationale == "" {
		return promptChoice{}, fmt.Errorf("%w: empty required field", ErrPromptInvalidOutput)
	}
	return choice, nil
}

func promptCandidateByModel(
	candidates []config.ModelRef,
	model string,
) *config.ModelRef {
	for i := range candidates {
		if candidates[i].Model == model {
			return &candidates[i]
		}
	}
	return nil
}

func (s *PromptSelector) selectionResult(
	candidates []config.ModelRef,
	selected config.ModelRef,
	rationale string,
	invocation PromptInvocationResult,
) *SelectionResult {
	return &SelectionResult{
		SelectedModel:          selected.Model,
		LoRAName:               selected.LoRAName,
		Score:                  1.0,
		Confidence:             1.0,
		Method:                 MethodPrompt,
		Tier:                   s.Tier(),
		Reasoning:              rationale,
		AllScores:              promptSelectionScores(candidates, selected.Model),
		HelperModel:            invocation.Model,
		HelperPromptTokens:     invocation.PromptTokens,
		HelperCompletionTokens: invocation.CompletionTokens,
		HelperTotalTokens:      invocation.TotalTokens,
		HelperLatencyMs:        invocation.LatencyMs,
	}
}

func (s *PromptSelector) systemPrompt(candidates []config.ModelRef) string {
	var builder strings.Builder
	builder.WriteString(strings.TrimSpace(s.config.Instructions))
	builder.WriteString("\n\nChoose exactly one candidate from this list:\n")
	for _, candidate := range candidates {
		builder.WriteString("- ")
		builder.WriteString(candidate.Model)
		if description := strings.TrimSpace(s.descriptions[candidate.Model]); description != "" {
			builder.WriteString(": ")
			builder.WriteString(description)
		}
		builder.WriteString("\n")
	}
	builder.WriteString(
		`Return only JSON: {"selected_model":"<exact candidate name>","rationale":"<short reason>"}.`,
	)
	return builder.String()
}

func promptSelectionScores(candidates []config.ModelRef, selected string) map[string]float64 {
	scores := make(map[string]float64, len(candidates))
	for _, candidate := range candidates {
		if candidate.Model == selected {
			scores[candidate.Model] = 1.0
		} else {
			scores[candidate.Model] = 0.0
		}
	}
	return scores
}
