package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// pluginPreviewContext creates private request state bound to exactly one
// active recipe. It never falls back to another recipe's decision/classifier.
func (r *OpenAIRouter) pluginPreviewContext(ctx context.Context, binding pluginruntime.Binding, pluginType string) (*RequestContext, error) {
	if r == nil || r.Config == nil {
		return nil, pluginruntime.ErrUnavailable
	}
	if binding.Recipe == "" || binding.Decision == "" {
		return nil, pluginruntime.ErrInvalidBinding
	}
	var recipe *config.RoutingRecipe
	if r.Config.IsDefaultRecipe(binding.Recipe) && len(r.Config.Recipes) == 0 {
		recipe = r.Config.DefaultRecipe()
	} else {
		recipe, _ = r.Config.RecipeByName(binding.Recipe)
	}
	if recipe == nil {
		return nil, pluginruntime.ErrInvalidBinding
	}
	for i := range recipe.Profile.Decisions {
		decision := &recipe.Profile.Decisions[i]
		if decision.Name != binding.Decision || decision.GetPlugin(pluginType) == nil {
			continue
		}
		request := &RequestContext{TraceContext: ctx, VSRSelectedDecision: decision, VSRSelectedDecisionName: decision.Name}
		request.Routing.SelectRecipe(recipe)
		return request, nil
	}
	return nil, pluginruntime.ErrInvalidBinding
}

// PreviewResponseGuard reads an active binding's response_guard over one
// answer, as the response path would: the guard's request-stage conditions on
// the question, then its rules in order over the answer and the context.
// Only mode=probe asks the configured models.
func (r *OpenAIRouter) PreviewResponseGuard(ctx context.Context, input pluginruntime.ResponseGuardPreviewRequest) (pluginruntime.ResponseGuardPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.ResponseGuardPreviewResponse{}, err
	}
	request, err := r.pluginPreviewContext(ctx, input.Binding, config.DecisionPluginResponseGuard)
	if err != nil {
		return pluginruntime.ResponseGuardPreviewResponse{}, err
	}
	result := pluginruntime.ResponseGuardPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding, Action: config.ResponseGuardNone}
	if strings.TrimSpace(input.Response) == "" {
		result.Reason = "empty_response"
		return result, nil
	}
	if mode != pluginruntime.ModeProbe {
		result.Reason = "probe_required"
		return result, nil
	}
	classifier := r.classifierForRequest(request)
	if classifier == nil {
		return result, pluginruntime.ErrUnavailable
	}
	result.BackendCalls = true
	if strings.TrimSpace(input.Question) != "" {
		request.RoutedSignals = classifier.EvaluateAllSignals(input.Question).Matches()
	}
	request.ResponseStage = classification.ResponseInput{Request: input.Question}
	request.RAGRetrievedContext = input.Context
	run := r.runResponseGuard(request, input.Response)
	if blocking := run.blocking(); blocking != nil {
		result.Action = config.ResponseGuardBlock
	} else if result.Warnings = run.warnings(); len(result.Warnings) > 0 {
		result.Action = config.ResponseGuardAnnotate
	}
	for _, rule := range run.rules {
		outcome := pluginruntime.ResponseGuardRuleOutcome{State: ruleOutcomeName(rule.state), Action: rule.action, MatchedRules: rule.matchedRules}
		if rule.action == config.ResponseGuardAnnotate {
			outcome.Warnings = rule.codes
		}
		result.Rules = append(result.Rules, outcome)
	}
	for _, outcome := range run.replayOutcomes(request, time.Now().UTC()) {
		if outcome.Target == config.DecisionPluginResponseGuard {
			continue
		}
		signal := pluginruntime.ResponseGuardSignalOutcome{Signal: outcome.Target, Verdict: outcome.Verdict, Reason: outcome.Reason}
		if outcome.Metadata["score_available"] == "true" {
			score := outcome.Score
			signal.Score = &score
		}
		if strings.HasPrefix(outcome.Target, config.SignalTypeHallucination+":") {
			for _, span := range run.observed().Spans[outcome.Target] {
				signal.UnsupportedSpans = append(signal.UnsupportedSpans, span.Text)
			}
		}
		result.Signals = append(result.Signals, signal)
	}
	return result, nil
}
