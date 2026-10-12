package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

// topicContinuityRules returns the rules declared by the recipe selected for
// this request, read through its classifier like every other rule, so one
// entrypoint's rules never run for another entrypoint's traffic.
func (r *OpenAIRouter) topicContinuityRules(ctx *RequestContext) []config.TopicContinuityRule {
	cfg := classifierConfig(r.classifierForRequest(ctx))
	if cfg == nil {
		return nil
	}
	return cfg.TopicContinuityRules
}

// evaluateTopicContinuity evaluates every topic-continuity rule the selected
// recipe declares, once per provider-bound request, immediately before the
// context transformation stage. It is driven by the declared rules, not by a
// consumer: the evidence is published whether or not a context policy reads
// it. It reads only the original (pre-enrichment) history and never mutates
// the request.
func (r *OpenAIRouter) evaluateTopicContinuity(ctx *RequestContext) {
	if ctx.TopicContinuityResults != nil {
		return
	}
	rules := r.topicContinuityRules(ctx)
	if len(rules) == 0 {
		return
	}
	configs := make([]topiccontinuity.EvalConfig, 0, len(rules))
	for _, rule := range rules {
		configs = append(configs, rule.EvalConfig())
	}
	load := func() ([]llmprotocol.Message, bool) {
		history, ok := originalConversation(ctx)
		return history.Messages, ok
	}
	evaluation := topiccontinuity.EvaluateAll(ctx.TraceContext, load, configs)
	ctx.TopicContinuityEvaluations = evaluation.Rules
	ctx.TopicContinuityResults = make(map[string]topiccontinuity.Result, len(evaluation.Rules))
	for _, rule := range evaluation.Rules {
		ctx.TopicContinuityResults[rule.Result.Signal] = rule.Result
	}
	recordTopicContinuityEvaluation(ctx, evaluation)
}

// recordTopicContinuityEvaluation records content-free metrics and one log
// event per rule, in declaration order. Internal errors get their own event so
// a faulting rule is visible without carrying content.
func recordTopicContinuityEvaluation(ctx *RequestContext, evaluation topiccontinuity.Evaluation) {
	for _, inputBytes := range evaluation.PreparationInputBytes {
		metrics.RecordTopicContinuityPreparation(inputBytes)
	}
	for _, rule := range evaluation.Rules {
		result := rule.Result
		metrics.RecordTopicContinuityEvaluation(string(result.Class), string(result.Reason),
			string(result.Coverage), result.Fallback)
		fields := map[string]interface{}{
			"request_id":         ctx.RequestID,
			"signal":             result.Signal,
			"schema_version":     result.SchemaVersion,
			"evaluator_version":  result.EvaluatorVersion,
			"class":              string(result.Class),
			"confidence":         result.Confidence,
			"reason":             string(result.Reason),
			"fallback":           result.Fallback,
			"coverage":           string(result.Coverage),
			"prior_turns":        result.Features.PriorTurnsExamined,
			"input_bytes":        result.Features.InputBytes,
			"classify_us":        rule.ClassifyMicros,
			"prepare_extract_us": rule.PrepareExtractMicros,
		}
		logging.ComponentEvent("extproc", "topic_continuity_evaluated", fields)
		if result.Reason == topiccontinuity.ReasonInternalError {
			logging.ComponentWarnEvent("extproc", "topic_continuity_internal_error", map[string]interface{}{
				"request_id": ctx.RequestID,
				"signal":     result.Signal,
				"reason":     string(result.Reason),
			})
		}
	}
}
