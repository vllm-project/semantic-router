package extproc

import (
	"context"
	"fmt"
	"strings"

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
	if binding.Recipe == config.DefaultRecipeName && len(r.Config.Recipes) == 0 {
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

func (r *OpenAIRouter) PreviewResponseJailbreak(ctx context.Context, input pluginruntime.ResponseJailbreakPreviewRequest) (pluginruntime.GuardPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.GuardPreviewResponse{}, err
	}
	request, err := r.pluginPreviewContext(ctx, input.Binding, config.DecisionPluginResponseJailbreak)
	if err != nil {
		return pluginruntime.GuardPreviewResponse{}, err
	}
	policy := request.VSRSelectedDecision.GetResponseJailbreakConfig()
	result := pluginruntime.GuardPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding, Enabled: policy.Enabled, Action: "none", DetectionSource: "plugin"}
	if !policy.Enabled {
		result.Reason = "disabled"
		return result, nil
	}
	if strings.TrimSpace(input.Response) == "" {
		result.Reason = "empty_response"
		return result, nil
	}
	result.Eligible = true
	if mode != pluginruntime.ModeProbe {
		result.Reason = "probe_required"
		return result, nil
	}
	classifier := r.classifierForRequest(request)
	if classifier == nil || !classifier.IsJailbreakEnabled() {
		return result, pluginruntime.ErrUnavailable
	}
	result.BackendCalls = true
	rules := r.responseJailbreakRules(request)
	if len(rules) > 0 {
		result.DetectionSource = "response_signal"
		scan, scanErr := classifier.ScanJailbreak(ctx, input.Response)
		var observed *classification.JailbreakScan
		if scanErr == nil {
			observed = &scan
			result.Label = scan.Type
			if scan.Decision == nil {
				result.Score = &scan.RiskScore
				result.ScoreKind = "probability"
			}
		}
		signal := classification.EvaluateResponseJailbreakSignal(rules, observed)
		result.MatchedRules = signal.MatchedRules
		result.Detected = len(signal.MatchedRules) > 0
		result.Resolved = result.Detected || len(signal.Errors) == 0
	} else {
		verdict, inferErr := classifier.CheckForJailbreakVerdict(ctx, input.Response, r.responseJailbreakThreshold(request.VSRSelectedDecision))
		if inferErr == nil {
			result.Resolved = true
			result.Detected = verdict.Detected
			result.Label = verdict.Label
			result.Score = verdict.RiskScore
			if result.Score != nil {
				result.ScoreKind = "probability"
			}
		}
	}
	if !result.Resolved {
		result.Reason = "detection_failed"
		result.Detected = responseJailbreakFailsClosed(classifierConfig(classifier))
	}
	if result.Detected {
		result.Action = r.getResponseJailbreakAction(request.VSRSelectedDecision)
	}
	return result, nil
}

func (r *OpenAIRouter) PreviewHallucination(ctx context.Context, input pluginruntime.HallucinationPreviewRequest) (pluginruntime.GuardPreviewResponse, error) {
	mode, err := pluginruntime.NormalizeMode(input.Mode)
	if err != nil {
		return pluginruntime.GuardPreviewResponse{}, err
	}
	request, err := r.pluginPreviewContext(ctx, input.Binding, config.DecisionPluginHallucination)
	if err != nil {
		return pluginruntime.GuardPreviewResponse{}, err
	}
	policy := request.VSRSelectedDecision.GetHallucinationConfig()
	result := pluginruntime.GuardPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: mode}, Binding: input.Binding, Enabled: policy.Enabled, Action: "none", DetectionSource: "plugin"}
	if !policy.Enabled {
		result.Reason = "disabled"
		return result, nil
	}
	if !input.FactCheckNeeded {
		result.Reason = "fact_check_not_needed"
		return result, nil
	}
	request.FactCheckNeeded = true
	request.UserContent = input.Question
	request.ToolResultsContext = input.Context
	request.HasToolsForFactCheck = strings.TrimSpace(input.Context) != ""
	if !request.HasToolsForFactCheck {
		result.Reason = "grounding_context_unavailable"
		result.Action = r.getUnverifiedFactualActionForDecision(request.VSRSelectedDecision)
		return result, nil
	}
	if strings.TrimSpace(input.Response) == "" {
		result.Reason = "empty_response"
		return result, nil
	}
	result.Eligible = true
	if mode != pluginruntime.ModeProbe {
		result.Reason = "probe_required"
		return result, nil
	}
	classifier := r.classifierForRequest(request)
	if classifier == nil || !classifier.IsHallucinationDetectionEnabled() {
		return result, pluginruntime.ErrUnavailable
	}
	useNLI := policy.UseNLI
	rules := r.hallucinationRules(request)
	if len(rules) > 0 {
		result.DetectionSource = "response_signal"
		useNLI = hallucinationRulesUseNLI(rules)
	}
	result.BackendCalls = true
	evidence, err := r.detectHallucinationEvidence(classifier, request, input.Response, useNLI)
	if err != nil {
		result.Reason = "detection_failed"
		return result, nil
	}
	if evidence == nil {
		return result, fmt.Errorf("%w: hallucination detector returned no evidence", pluginruntime.ErrUnavailable)
	}
	result.Resolved = true
	result.Detected = evidence.Detected
	result.UnsupportedSpans = append([]string(nil), evidence.Spans...)
	if evidence.ScoreAvailable {
		result.Score = &evidence.Confidence
		result.ScoreKind = evidence.ScoreKind
	}
	if len(rules) > 0 {
		signal := classification.EvaluateResponseHallucinationSignal(rules, evidence.Detected, evidence.Confidence, "", classification.HallucinationScore{Available: evidence.ScoreAvailable, Kind: evidence.ScoreKind})
		result.MatchedRules = signal.MatchedRules
	}
	if result.Detected {
		result.Action = r.getHallucinationActionForDecision(request.VSRSelectedDecision)
	}
	return result, nil
}
