package extproc

import (
	"fmt"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

func (r *OpenAIRouter) applySemanticReasoningMode(
	request *llmprotocol.Request,
	enabled bool,
	decision *config.Decision,
) bool {
	if request == nil || !enabled || r.getModelReasoningFamily(request.Model) == nil {
		return false
	}
	effort := r.getReasoningEffort(decision, request.Model)
	if request.ReasoningEffort == effort {
		return false
	}
	request.ReasoningEffort = effort
	logging.Infof("Applied reasoning effort %q to model %q", effort, request.Model)
	return true
}

func (r *OpenAIRouter) addSemanticSystemPromptIfConfigured(
	request *llmprotocol.Request,
	decisionName string,
	model string,
	ctx *RequestContext,
) (bool, error) {
	if request == nil || decisionName == "" || ctx == nil || ctx.VSRSelectedDecision == nil {
		return false, nil
	}
	decision := ctx.VSRSelectedDecision
	promptConfig := decision.GetSystemPromptConfig()
	if promptConfig == nil || promptConfig.SystemPrompt == "" || !decision.IsSystemPromptEnabled() {
		return false, nil
	}
	start := time.Now()
	promptContext, span := tracing.StartPluginSpan(ctx.TraceContext, "system_prompt", decisionName)
	mode := decision.GetSystemPromptMode()
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: promptConfig.SystemPrompt}
	injected := false
	for index := range request.Instructions {
		instruction := &request.Instructions[index]
		if instruction.Role != llmprotocol.RoleSystem {
			continue
		}
		if mode == "insert" {
			instruction.Content = append([]llmprotocol.Content{content}, instruction.Content...)
		} else {
			instruction.Content = []llmprotocol.Content{content}
		}
		injected = true
		break
	}
	if !injected {
		request.Instructions = append([]llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{content},
		}}, request.Instructions...)
		injected = true
	}
	latency := time.Since(start).Milliseconds()
	tracing.SetSpanAttributes(span,
		attribute.Bool("system_prompt.injected", injected),
		attribute.String("system_prompt.mode", mode),
		attribute.String(tracing.AttrCategoryName, decisionName),
	)
	tracing.EndPluginSpan(span, "success", latency, "prompt_injected")
	ctx.TraceContext = promptContext
	ctx.VSRInjectedSystemPrompt = true
	logging.Infof("Applied system instruction for decision %q to model %q", decisionName, model)
	return true, nil
}

func (r *OpenAIRouter) applySemanticRequestParams(
	decision *config.Decision,
	request *llmprotocol.Request,
	routingScope config.RecipeName,
) (bool, error) {
	if decision == nil || request == nil || decision.GetRequestParamsConfig() == nil {
		return false, nil
	}
	params := decision.GetRequestParamsConfig()
	decisionKey := config.RoutingDecisionKey(routingScope, decision.Name)
	changed := false
	for _, field := range params.BlockedParams {
		blocked, err := blockSemanticRequestField(request, strings.TrimSpace(field))
		if err != nil {
			return false, err
		}
		if blocked {
			changed = true
			metrics.RecordBlockedParam(decisionKey, field)
		}
	}
	if params.MaxTokensLimit != nil && request.Sampling.MaxOutputTokens != nil &&
		*request.Sampling.MaxOutputTokens > int64(*params.MaxTokensLimit) {
		request.Sampling.MaxOutputTokens = llmprotocol.Int64(int64(*params.MaxTokensLimit))
		metrics.RecordMaxTokensCapped(decisionKey)
		changed = true
	}
	if params.MaxN != nil && request.CandidateCount != nil &&
		*request.CandidateCount > int64(*params.MaxN) {
		request.CandidateCount = llmprotocol.Int64(int64(*params.MaxN))
		metrics.RecordMaxNCapped(decisionKey)
		changed = true
	}
	return changed, nil
}

func blockSemanticRequestField(request *llmprotocol.Request, field string) (bool, error) {
	switch field {
	case "":
		return false, nil
	case "model", "messages":
		return false, fmt.Errorf("required semantic field %q cannot be blocked", field)
	case "frequency_penalty":
		changed := request.Sampling.FrequencyPenalty != nil
		request.Sampling.FrequencyPenalty = nil
		return changed, nil
	case "presence_penalty":
		changed := request.Sampling.PresencePenalty != nil
		request.Sampling.PresencePenalty = nil
		return changed, nil
	case "max_tokens", "max_completion_tokens", "max_output_tokens":
		changed := request.Sampling.MaxOutputTokens != nil
		request.Sampling.MaxOutputTokens = nil
		return changed, nil
	case "n", "candidate_count":
		changed := request.CandidateCount != nil
		request.CandidateCount = nil
		return changed, nil
	case "response_format", "output_format":
		changed := request.OutputFormat.Kind != ""
		request.OutputFormat = llmprotocol.OutputFormat{}
		return changed, nil
	case "seed":
		changed := request.Sampling.Seed != nil
		request.Sampling.Seed = nil
		return changed, nil
	case "stop":
		changed := len(request.Sampling.Stop) > 0
		request.Sampling.Stop = nil
		return changed, nil
	case "temperature":
		changed := request.Sampling.Temperature != nil
		request.Sampling.Temperature = nil
		return changed, nil
	case "top_p":
		changed := request.Sampling.TopP != nil
		request.Sampling.TopP = nil
		return changed, nil
	case "top_k":
		changed := request.Sampling.TopK != nil
		request.Sampling.TopK = nil
		return changed, nil
	case "tools":
		changed := len(request.Tools) > 0
		request.Tools = nil
		return changed, nil
	case "tool_choice":
		changed := request.ToolChoice.Mode != "" || request.ToolChoice.Name != ""
		request.ToolChoice = llmprotocol.ToolChoice{}
		return changed, nil
	case "parallel_tool_calls":
		changed := request.ParallelToolCalls != nil
		request.ParallelToolCalls = nil
		return changed, nil
	case "reasoning_effort":
		changed := request.ReasoningEffort != ""
		request.ReasoningEffort = ""
		return changed, nil
	case "reasoning_budget_tokens":
		changed := request.ReasoningBudgetTokens != nil
		request.ReasoningBudgetTokens = nil
		return changed, nil
	case "metadata":
		changed := len(request.Metadata) > 0
		request.Metadata = nil
		return changed, nil
	case "store":
		changed := request.Store != nil
		request.Store = nil
		return changed, nil
	case "stream":
		changed := request.Stream
		request.Stream = false
		return changed, nil
	default:
		// Unknown client fields never enter neutral IR; codecs already enforce
		// the configured unknown-field policy at ingress.
		return false, nil
	}
}
