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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// A decision model serves the PII and hallucination bindings through
// /v1/decisions instead of a classify head: it answers the consumer's span
// preset, as the model's own ready-made question when its card defines it
// (the pii and halu questions the Vela 2.0 and Vela 3.0 families were
// trained on, answered by their router span head; presets.Binding
// ModelPreset), else as the preset's canonical span question. The binding
// reads the spans as token_spans.v1, so its consumer is unchanged. In a
// request stage the question travels in the same call as the stage's other
// questions to the deployment (see modelservice.Bundle).

// spanTask is the span preset a consumer asks, in its stage, the built-in
// task that asks it and the ready-made question a model defines for it.
type spanTask struct {
	preset, stage, task, modelPreset string
}

var spanTasks = map[string]spanTask{
	"pii_classifier":         {presets.PII, presets.StageRequest, "pii_spans", "pii"},
	"hallucination_detector": {presets.Hallucination, presets.StageResponse, "hallucination_spans", "halu"},
}

// spanQuestion is a binding's model preset and the ID its question carries
// in a decisions call: the consumer's name, a ':' and the preset, never a
// systemone rule's ID (systemone:<rule>).
func spanQuestion(spec config.ResolvedModelBinding, preset string) modelservice.Question {
	return modelservice.Question{ID: spec.Name + ":" + preset, Preset: preset}
}

// spanPreset reports whether a binding asks a span preset of a deployment
// whose model answers decisions rather than classify heads, and the model's
// own preset question it asks, empty for the canonical one.
func spanPreset(spec config.ResolvedModelBinding, card modelservice.ModelCard) (string, bool) {
	task, ok := spanTasks[spec.Name]
	if !ok || !card.Serves("decisions") || (spec.Binding.Head != "" && spec.Binding.Head != config.SystemOneSpanHeadRouter) {
		return "", false
	}
	if card.HasPreset(task.modelPreset) {
		return task.modelPreset, true
	}
	return "", true
}

// prepareSpanQuestion checks a span binding against its deployment's card:
// the model must answer span questions or define the preset its family was
// trained on, which its router span head answers, and a decision model reads
// its whole input, so the binding sets no head other than router, no label
// mapping and, on a declared deployment, no input budget.
func (r *Runtime) prepareSpanQuestion(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*target, binding.Capability, modelservice.Decider, error) {
	decider, ok := r.services.(modelservice.Decider)
	if !ok {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: the model runtime services of deployment %q answer no decisions", binding.ErrCapability, spec.Binding.Deployment)
	}
	task := spanTasks[spec.Name]
	if preset == "" && !card.Answers("span") {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: deployment %q serves %s, which answers no span questions and defines no ready-made %s question; bind a token head or a span-capable model", binding.ErrCapability, spec.Binding.Deployment, card.ID, task.preset)
	}
	if spec.Binding.MappingPath != "" {
		return nil, binding.Capability{}, nil, fmt.Errorf("%w: the %s question of deployment %q names its own labels; remove mapping_path", binding.ErrCapability, task.preset, spec.Binding.Deployment)
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
		Contract: spec.Binding.Contract, Provider: providerOf(spec), Device: card.Device, Precision: card.Dtype, Preset: preset,
		Deployment: spec.Binding.Deployment,
		Limits:     binding.Limits{ModelTokens: card.MaxInputTokens, Overflow: spec.Deployment.Input.Overflow},
	}
	return &target{spec: spec, deployment: spec.Binding.Deployment, card: card, resource: resource, scan: scan}, capability, decider, nil
}

// askSpans asks one span question and returns its spans in text order.
func askSpans(ctx context.Context, decider modelservice.Decider, deployment string, request modelservice.Request, cards ...modelservice.ModelCard) ([]modelservice.Span, error) {
	question := request.Questions[0]
	var response modelservice.Response
	var err error
	if len(cards) > 0 {
		response, err = modelservice.ExecuteQuestions(ctx, decider, deployment, cards[0], request)
	} else {
		response, err = decider.Decide(ctx, deployment, request)
	}
	if err != nil {
		return nil, err
	}
	answer, ok := response.Answers[question.ID]
	switch {
	case !ok:
		return nil, fmt.Errorf("%w: the %s question has no answer", binding.ErrInvalidResult, question.Preset)
	case answer.Error != "":
		return nil, itemError(answer.Error)
	case answer.Type != config.SystemOneQuestionSpan:
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

// spanBindingQuestion is the question a span binding asks: the model's own
// preset question, or the canonical question of the consumer's span preset
// under the consumer's ID.
func spanBindingQuestion(spec config.ResolvedModelBinding, preset string, capability *binding.Capability) modelservice.Question {
	task := spanTasks[spec.Name]
	question := spanQuestion(spec, preset)
	if preset == "" {
		definition, _ := modelservice.BuiltinTask(task.task)
		question = definition.Question
		question.ID = spec.Name + ":" + task.task
		capability.Preset, capability.Question = "", task.task
	}
	question.TaskID, question.Stage = task.task, task.stage
	return question
}

// spanTokens binds a text span consumer (PII) to the deployment's span
// question.
func (r *Runtime) spanTokens(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*binding.Resolved[string, tasks.TokenClassificationResult], error) {
	t, capability, decider, err := r.prepareSpanQuestion(ctx, spec, card, preset)
	if err != nil {
		return nil, err
	}
	question := spanBindingQuestion(spec, preset, &capability)
	return publish(ctx, r.tokens, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.TokenClassificationResult, error) {
		spans, err := askSpans(ctx, decider, t.deployment, readPolicy(modelservice.Request{State: text, Questions: []modelservice.Question{question}}, true, t.scan), card)
		if err != nil {
			return tasks.TokenClassificationResult{}, err
		}
		return spanResult(text, spans)
	}, "warmup")
}

// spanGrounded binds the hallucination consumer to the deployment's span
// question: the state's request, context and answer are the question, the
// grounding context and the answer, and the spans refer to the answer.
func (r *Runtime) spanGrounded(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, preset string) (*binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult], error) {
	t, capability, decider, err := r.prepareSpanQuestion(ctx, spec, card, preset)
	if err != nil {
		return nil, err
	}
	question := spanBindingQuestion(spec, preset, &capability)
	warmup := tasks.GroundedTextRequest{Context: "A warmup sentence.", Question: "What is this?", Answer: "A sentence."}
	return publish(ctx, r.grounded, t, capability, func(ctx context.Context, _ io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
		parts := map[string]string{"context": input.Context, "answer": input.Answer}
		if strings.TrimSpace(input.Question) != "" {
			parts["request"] = input.Question
		}
		spans, err := askSpans(ctx, decider, t.deployment, readPolicy(modelservice.Request{Parts: parts, Questions: []modelservice.Question{question}}, true, t.scan), card)
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
