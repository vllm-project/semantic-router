package serving

import (
	"context"
	"fmt"
	"io"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// classifyOptions are the request options a prepared binding sends with every call.
type classifyOptions struct {
	overflow  string
	maxTokens int
	window    *modelservice.Window
	threshold *float64
}

// singleOptions applies the deployment's budget to one forward per input.
func singleOptions(spec config.ResolvedModelBinding) classifyOptions {
	return classifyOptions{overflow: spec.Deployment.Input.Overflow, maxTokens: spec.Deployment.Input.MaxTokens}
}

// rejectWindowPolicy refuses a head binding whose deployment reads in windows:
// a head reads windows only through its typed window task. (A deployment
// that answers questions takes window as its scan budget instead.)
func rejectWindowPolicy(spec config.ResolvedModelBinding, task string) error {
	if spec.Deployment.Input.Overflow == "window" {
		return fmt.Errorf("%w: %s window policy requires the typed window task", binding.ErrCapability, task)
	}
	return nil
}

// classify runs one input through the target's head. The response must use
// the prepared head's label order; an item error maps to the binding errors
// that consumers' fail-open policies already know.
func (r *Runtime) classify(ctx context.Context, t *target, input modelservice.ClassifyInput, options classifyOptions) (modelservice.ClassifyResult, error) {
	response, err := r.services.Classify(ctx, t.deployment, modelservice.ClassifyRequest{
		Head: t.head.Name, Inputs: []modelservice.ClassifyInput{input}, Overflow: options.overflow,
		MaxTokens: options.maxTokens, Window: options.window, Threshold: options.threshold,
	})
	if err != nil {
		return modelservice.ClassifyResult{}, err
	}
	if len(response.Results) != 1 || !slices.Equal(response.Labels, t.head.Labels) {
		return modelservice.ClassifyResult{}, fmt.Errorf("%w: classify response does not match the prepared head", binding.ErrInvalidResult)
	}
	result := response.Results[0]
	if result.Error != "" {
		return result, itemError(result.Error)
	}
	return result, nil
}

func itemError(code string) error {
	switch code {
	case "max_length_exceeded":
		return fmt.Errorf("%w: %s", binding.ErrInputLimit, code)
	case "scan_budget_exceeded":
		return fmt.Errorf("%w: %s", binding.ErrScanBudget, code)
	case "invalid_input":
		return fmt.Errorf("%w: %s", binding.ErrInvalidInput, code)
	case "invalid_model_output":
		return fmt.Errorf("%w: %s", binding.ErrInvalidResult, code)
	case "deadline_exceeded":
		return context.DeadlineExceeded
	case "unavailable":
		return modelservice.ErrUnavailable
	default:
		return fmt.Errorf("%w: %s", modelservice.ErrFailed, code)
	}
}

// Sequence binds a categorical head: one label distribution per input. A
// built-in signal bound to a Vela 2.0 model asks it the signal's question
// instead.
func (r *Runtime) Sequence(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.LabelDistribution], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if question, ok := questionFor(spec, card); ok {
		return r.questionSequence(ctx, spec, card, question)
	}
	if err = rejectWindowPolicy(spec, "sequence"); err != nil {
		return nil, err
	}
	t, capability, err := r.prepareHead(ctx, spec, kindSequence, inputText)
	if err != nil {
		return nil, err
	}
	options := singleOptions(spec)
	return publish(ctx, r.sequence, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.LabelDistribution, error) {
		result, err := r.classify(ctx, t, modelservice.ClassifyInput{Text: text}, options)
		if err != nil {
			return tasks.LabelDistribution{}, err
		}
		return tasks.LabelDistribution{Probabilities: float32s(result.Probabilities), Input: inputUsage(result.Input)}, nil
	}, "warmup")
}

// Scores binds an independent-label head: one sigmoid score per label.
func (r *Runtime) Scores(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.LabelScores], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if err := rejectWindowPolicy(spec, "label score"); err != nil {
		return nil, err
	}
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	t, capability, err := r.prepareHead(ctx, spec, kindScores, inputText)
	if err != nil {
		return nil, err
	}
	options := singleOptions(spec)
	return publish(ctx, r.scores, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.LabelScores, error) {
		result, err := r.classify(ctx, t, modelservice.ClassifyInput{Text: text}, options)
		if err != nil {
			return tasks.LabelScores{}, err
		}
		return tasks.LabelScores{Scores: float32s(result.Scores), Input: inputUsage(result.Input)}, nil
	}, "warmup")
}

// Tokens binds a token head; spans arrive in code points and leave in UTF-8
// byte offsets. A truncated input keeps its valid spans and reports
// tasks.ErrTokenSpansTruncated so the consumer applies its partial-input policy.
// A PII binding to a decision model asks its ready-made pii question instead.
func (r *Runtime) Tokens(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.TokenClassificationResult], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if preset, ok := spanPreset(spec, card); ok {
		return r.spanTokens(ctx, spec, card, preset)
	}
	if err = rejectWindowPolicy(spec, "token"); err != nil {
		return nil, err
	}
	t, capability, err := r.prepareHead(ctx, spec, kindToken, inputText)
	if err != nil {
		return nil, err
	}
	options := singleOptions(spec)
	return publish(ctx, r.tokens, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.TokenClassificationResult, error) {
		result, err := r.classify(ctx, t, modelservice.ClassifyInput{Text: text}, options)
		if err != nil {
			return tasks.TokenClassificationResult{}, err
		}
		out, err := tokenResult(text, result)
		if err == nil && out.Input != nil && out.Input.Truncated {
			err = tasks.ErrTokenSpansTruncated
		}
		return out, err
	}, "warmup")
}

// Grounded binds a hallucination head: the answer is read against its
// context and question, and spans refer to the answer. A binding to a
// decision model asks its ready-made halu question at the model's own
// calibrated threshold instead, so threshold applies to heads only.
func (r *Runtime) Grounded(ctx context.Context, spec config.ResolvedModelBinding, threshold float32) (_ *binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if preset, ok := spanPreset(spec, card); ok {
		return r.spanGrounded(ctx, spec, card, preset)
	}
	if err = rejectWindowPolicy(spec, "grounded"); err != nil {
		return nil, err
	}
	t, capability, err := r.prepareHead(ctx, spec, kindToken, inputGrounded)
	if err != nil {
		return nil, err
	}
	options := singleOptions(spec)
	if threshold > 0 {
		value := float64(threshold)
		options.threshold = &value
	}
	warmup := tasks.GroundedTextRequest{Context: "A warmup sentence.", Question: "What is this?", Answer: "A sentence."}
	return publish(ctx, r.grounded, t, capability, func(ctx context.Context, _ io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
		result, err := r.classify(ctx, t, modelservice.ClassifyInput{Context: input.Context, Question: input.Question, Answer: input.Answer}, options)
		if err != nil {
			return tasks.TokenClassificationResult{}, err
		}
		out, err := tokenResult(input.Answer, result)
		if err != nil {
			return out, err
		}
		summarizeSpans(&out)
		return out, nil
	}, warmup)
}

func tokenResult(text string, result modelservice.ClassifyResult) (tasks.TokenClassificationResult, error) {
	entities, err := byteSpans(text, result.Spans)
	if err != nil {
		return tasks.TokenClassificationResult{}, err
	}
	scored := true
	return tasks.TokenClassificationResult{Input: inputUsage(result.Input), Entities: entities, ScoresAvailable: &scored}, nil
}

func inputUsage(usage *modelservice.InputUsage) *tasks.InputUsage {
	if usage == nil {
		return nil
	}
	return &tasks.InputUsage{OriginalTokens: usage.Tokens, ProcessedTokens: usage.ProcessedTokens, Truncated: usage.Truncated}
}

func float32s(values []float64) []float32 {
	if values == nil {
		return nil
	}
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}
