package serving

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A decision model that defines a ready-made span question (Vela 2.0's pii
// and halu presets, answered by its router span head) serves the PII and
// hallucination bindings through /v1/decisions instead of a classify head.
// The binding asks the preset and reads its spans as token_spans.v1, so its
// consumer is unchanged. In a request stage the question travels in the same
// call as the stage's other questions to the deployment (see
// modelservice.Bundle).

// spanPresets names the ready-made question each span consumer asks.
var spanPresets = map[string]string{
	"pii_classifier":         "pii",
	"hallucination_detector": "halu",
}

// spanQuestion is a binding's preset and the ID its question carries in a
// decisions call. A decision signal's name never contains ':', so the ID
// cannot collide with one of the signals asked in the same call.
func spanQuestion(spec config.ResolvedModelBinding, preset string) modelservice.Question {
	return modelservice.Question{ID: spec.Name + ":" + preset, Preset: preset}
}

// spanPreset returns the ready-made question a binding asks when its
// deployment's model answers decisions rather than classify heads.
func spanPreset(spec config.ResolvedModelBinding, card modelservice.ModelCard) (string, bool) {
	preset, ok := spanPresets[spec.Name]
	return preset, ok && card.Serves("decisions") && !card.Serves("classify")
}

// prepareSpanQuestion checks a span binding against its deployment's card:
// the model must define the preset, which its router span head answers, and
// a decision model reads its whole input, so the binding sets no head other
// than router, no label mapping and, on a declared deployment, no input
// budget.
func (r *Runtime) prepareSpanQuestion(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*target, binding.Capability, modelservice.Decider, error) {
	decider, ok := r.services.(modelservice.Decider)
	if !ok {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: the model runtime services of deployment %q answer no decisions", binding.ErrCapability, spec.Binding.Deployment)
	}
	if !card.HasPreset(preset) {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: deployment %q serves %s, which has no classify head and no ready-made %s question; bind a token head or a model that defines the %s question, such as Vela 2.0", binding.ErrCapability, spec.Binding.Deployment, card.ID, preset, preset)
	}
	if spec.Binding.Head != "" && spec.Binding.Head != config.DecisionSpanHeadRouter {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: the %s question of deployment %q is answered by its router span head; set head to router or remove it", binding.ErrCapability, preset, spec.Binding.Deployment)
	}
	if spec.Binding.MappingPath != "" {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: the %s question of deployment %q names its own labels; remove mapping_path", binding.ErrCapability, preset, spec.Binding.Deployment)
	}
	scan, err := questionScanBudget(spec, card)
	if err != nil {
		return nil, binding.Capability{}, nil, err
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, binding.Capability{}, nil, err
	}
	capability := binding.Capability{
		Contract: spec.Binding.Contract, Provider: Provider, Device: card.Device, Precision: card.Dtype, Preset: preset,
		Deployment: spec.Binding.Deployment,
		Limits:     binding.Limits{ModelTokens: card.MaxInputTokens, Overflow: spec.Deployment.Input.Overflow},
	}
	return &target{spec: spec, deployment: spec.Binding.Deployment, card: card, resource: resource, scan: scan}, capability, decider, nil
}

// askSpans asks one span question and returns its spans in text order.
func askSpans(ctx context.Context, decider modelservice.Decider, deployment string, request modelservice.Request) ([]modelservice.Span, error) {
	question := request.Questions[0]
	response, err := decider.Decide(ctx, deployment, request)
	if err != nil {
		return nil, err
	}
	answer, ok := response.Answers[question.ID]
	switch {
	case !ok:
		return nil, fmt.Errorf("%w: the %s question has no answer", binding.ErrInvalidResult, question.Preset)
	case answer.Error != "":
		return nil, itemError(answer.Error)
	case answer.Type != config.DecisionQuestionSpan:
		return nil, fmt.Errorf("%w: the %s question of deployment %q is a %s question, not a span question", binding.ErrCapability, question.Preset, deployment, answer.Type)
	}
	spans := append([]modelservice.Span(nil), answer.Spans...)
	sort.SliceStable(spans, func(i, j int) bool { return spans[i].Start < spans[j].Start })
	return spans, nil
}

func spanResult(text string, spans []modelservice.Span) (tasks.TokenClassificationResult, error) {
	entities, err := byteSpans(text, spans)
	if err != nil {
		return tasks.TokenClassificationResult{}, err
	}
	scored := true
	return tasks.TokenClassificationResult{Entities: entities, ScoresAvailable: &scored}, nil
}

// spanTokens binds a text span consumer (PII) to the deployment's preset.
func (r *Runtime) spanTokens(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*binding.Resolved[string, tasks.TokenClassificationResult], error) {
	t, capability, decider, err := r.prepareSpanQuestion(ctx, spec, card, preset)
	if err != nil {
		return nil, err
	}
	question := spanQuestion(spec, preset)
	return publish(ctx, r.tokens, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.TokenClassificationResult, error) {
		spans, err := askSpans(ctx, decider, t.deployment, readPolicy(modelservice.Request{State: text, Questions: []modelservice.Question{question}}, true, t.scan))
		if err != nil {
			return tasks.TokenClassificationResult{}, err
		}
		return spanResult(text, spans)
	}, "warmup")
}

// spanGrounded binds the hallucination consumer to the deployment's preset:
// the state's request, context and answer are the question, the grounding
// context and the answer, and the spans refer to the answer.
func (r *Runtime) spanGrounded(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult], error) {
	t, capability, decider, err := r.prepareSpanQuestion(ctx, spec, card, preset)
	if err != nil {
		return nil, err
	}
	question := spanQuestion(spec, preset)
	warmup := tasks.GroundedTextRequest{Context: "A warmup sentence.", Question: "What is this?", Answer: "A sentence."}
	return publish(ctx, r.grounded, t, capability, func(ctx context.Context, _ io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
		parts := map[string]string{"context": input.Context, "answer": input.Answer}
		if strings.TrimSpace(input.Question) != "" {
			parts["request"] = input.Question
		}
		spans, err := askSpans(ctx, decider, t.deployment, readPolicy(modelservice.Request{Parts: parts, Questions: []modelservice.Question{question}}, true, t.scan))
		if err != nil {
			return tasks.TokenClassificationResult{}, err
		}
		out, err := spanResult(input.Answer, spans)
		if err != nil {
			return out, err
		}
		summarizeSpans(&out)
		return out, nil
	}, warmup)
}

// summarizeSpans sets a span result's aggregate to its most probable span;
// an empty span set carries no model evidence, so it has none.
func summarizeSpans(out *tasks.TokenClassificationResult) {
	if len(out.Entities) == 0 {
		return
	}
	best := out.Entities[0].Confidence
	for _, entity := range out.Entities[1:] {
		best = max(best, entity.Confidence)
	}
	out.Summary = &tasks.ScoreResult{Value: float64(best)}
	out.SummarySemantics = &tasks.ScoreSemantics{Unit: "max_hallucinated_token_score", Direction: tasks.HigherIsPositive, Calibrated: false}
}
