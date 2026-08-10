package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type contextCompressionStats struct {
	appliedMessages int
	appliedBlocks   int
	beforeTokens    int
	afterTokens     int
	omittedChunks   int
	skippedRAG      int
	jsonBlocks      int
}

func (stats contextCompressionStats) format() string {
	switch stats.jsonBlocks {
	case 0:
		return "text"
	case stats.appliedBlocks:
		return "json"
	default:
		return "mixed"
	}
}

func (r *OpenAIRouter) applyContextCompression(
	ctx *RequestContext,
	requestBody []byte,
) []byte {
	compressed, err := r.applyContextCompressionPolicy(ctx, requestBody)
	if err != nil {
		return requestBody
	}
	return compressed
}

func (r *OpenAIRouter) applyContextCompressionPolicy(
	ctx *RequestContext,
	requestBody []byte,
) ([]byte, error) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return requestBody, nil
	}
	pluginConfig := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	if pluginConfig == nil || !pluginConfig.Enabled {
		return requestBody, nil
	}
	applyContextCompressionRequestControls(ctx, pluginConfig)
	if ctx.ContextCompressionSkipReason == "request_bypass" {
		recordContextCompressionStatus(ctx, "bypassed", 0)
		return requestBody, nil
	}

	start := time.Now()
	request, err := parseContextCompressionRequest(ctx, requestBody, start)
	if err != nil {
		return compressionFailure(pluginConfig, requestBody, err)
	}

	service := r.contextCompressionService()
	if service == nil {
		return compressionFailure(
			pluginConfig,
			requestBody,
			fmt.Errorf("context compression service is unavailable"),
		)
	}
	callContext, compressionRequest, ragProtected := r.buildContextCompressionRequest(
		ctx,
		pluginConfig,
		request,
	)
	result := service.Apply(callContext, compressionRequest)
	return r.finishContextCompression(
		ctx,
		pluginConfig,
		requestBody,
		request,
		result,
		ragProtected,
		start,
	)
}

func parseContextCompressionRequest(
	ctx *RequestContext,
	requestBody []byte,
	start time.Time,
) (map[string]interface{}, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		recordContextCompressionStatus(ctx, "parse_error_fail_open", time.Since(start).Seconds())
		return nil, err
	}
	if _, ok := request["messages"].([]interface{}); !ok {
		recordContextCompressionStatus(ctx, "unsupported_shape_fail_open", time.Since(start).Seconds())
		return nil, fmt.Errorf("context compression requires a messages array")
	}
	return request, nil
}

func (r *OpenAIRouter) buildContextCompressionRequest(
	ctx *RequestContext,
	pluginConfig *config.ContextCompressionPluginConfig,
	request map[string]interface{},
) (context.Context, contextcompression.Request, bool) {
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	policy := contextCompressionPolicy(
		pluginConfig,
		ctx.ContextCompressionTargetTokens,
	)
	policy = compressionPolicyForRequest(policy, ctx)
	ragProtected := requestContainsRAGToolResult(request) &&
		policy.Targets.RAG.Mode == contextcompression.TargetPreserve
	if revision, err := cache.FingerprintValue(pluginConfig); err == nil {
		ctx.ContextCompressionRevision = revision[:12]
	}
	requestIR := contextcompression.ParseRequestIR(
		request,
		contextcompression.Provenance{
			RAGToolCallIDs:       ctx.RAGToolCallIDs,
			MemoryMessageIndexes: ctx.MemoryMessageIndexes,
		},
	)
	callContext := ctx.TraceContext
	if callContext == nil {
		callContext = context.Background()
	}
	compressionRequest := contextcompression.Request{
		Model:        model,
		Scope:        r.contextCompressionScope(ctx),
		Request:      requestIR,
		Policy:       policy,
		Capabilities: contextCompressionCapabilities(r.Config, ctx),
		TokenCounter: r.contextCompressionTokenCounter(ctx),
		Scorer:       r.contextCompressionScorer(callContext, pluginConfig),
		Recovery:     r.contextCompressionRecoveryStore(pluginConfig),
		Provenance: contextcompression.Provenance{
			RAGToolCallIDs:       ctx.RAGToolCallIDs,
			MemoryMessageIndexes: ctx.MemoryMessageIndexes,
		},
	}
	return callContext, compressionRequest, ragProtected
}

func (r *OpenAIRouter) finishContextCompression(
	ctx *RequestContext,
	pluginConfig *config.ContextCompressionPluginConfig,
	requestBody []byte,
	request map[string]interface{},
	result contextcompression.ServiceResult,
	ragProtected bool,
	start time.Time,
) ([]byte, error) {
	if result.Failure != nil {
		recordContextCompressionStatus(ctx, "compression_failed", time.Since(start).Seconds())
		return compressionFailure(pluginConfig, requestBody, result.Failure)
	}
	r.recordCompressionPlan(ctx, result)
	if !result.Applied {
		status := string(result.Plan.SkipReason)
		if status == string(contextcompression.SkipNoTargets) && ragProtected {
			status = "rag_protected"
		}
		if status == "" {
			status = "not_applied"
		}
		recordContextCompressionStatus(ctx, status, time.Since(start).Seconds())
		return requestBody, nil
	}
	if len(result.RecoveryKeys) > 0 {
		if err := injectContextRecoveryTool(request, result.RecoveryKeys); err != nil {
			return compressionFailure(pluginConfig, requestBody, err)
		}
	}

	compressedBody, err := result.Request.Marshal()
	if err != nil {
		recordContextCompressionStatus(ctx, "marshal_error_fail_open", time.Since(start).Seconds())
		return compressionFailure(pluginConfig, requestBody, err)
	}
	stats := contextCompressionStats{
		appliedMessages: result.MessagesCompressed,
		appliedBlocks:   result.BlocksCompressed,
		beforeTokens:    result.TokensBefore,
		afterTokens:     result.TokensAfter,
		omittedChunks:   result.OmittedChunks,
		jsonBlocks:      result.JSONBlocks,
	}
	ctx.ContextCompressionRecoveryKeys = append(
		ctx.ContextCompressionRecoveryKeys[:0],
		result.RecoveryKeys...,
	)
	recordContextCompressionApplied(ctx, stats, start)
	return compressedBody, nil
}

func requestContainsRAGToolResult(request map[string]interface{}) bool {
	messages, _ := request["messages"].([]interface{})
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		if isRAGToolResult(message) {
			return true
		}
		blocks, _ := message["content"].([]interface{})
		for _, rawBlock := range blocks {
			block, ok := rawBlock.(map[string]interface{})
			if ok && isRAGToolResult(block) {
				return true
			}
		}
	}
	return false
}

func injectContextRecoveryTool(
	request map[string]interface{},
	keys []string,
) error {
	rawTools := request["tools"]
	tools := make([]interface{}, 0)
	if rawTools != nil {
		existing, ok := rawTools.([]interface{})
		if !ok {
			return fmt.Errorf("tools must be an array for recoverable compression")
		}
		tools = append(tools, existing...)
		for _, rawTool := range existing {
			tool, _ := rawTool.(map[string]interface{})
			function, _ := tool["function"].(map[string]interface{})
			if function["name"] == contextcompression.RetrieveToolName {
				return fmt.Errorf(
					"request defines reserved tool %q",
					contextcompression.RetrieveToolName,
				)
			}
		}
	}
	enumValues := make([]interface{}, len(keys))
	for index, key := range keys {
		enumValues[index] = key
	}
	tools = append(tools, map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        contextcompression.RetrieveToolName,
			"description": "Retrieve original context omitted by vLLM Semantic Router compression.",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"key": map[string]interface{}{
						"type": "string",
						"enum": enumValues,
					},
				},
				"required": []interface{}{"key"},
			},
		},
	})
	request["tools"] = tools
	return nil
}

func compressionFailure(
	plugin *config.ContextCompressionPluginConfig,
	requestBody []byte,
	err error,
) ([]byte, error) {
	if plugin.EffectiveFailureMode() == config.ContextCompressionFailureClosed {
		return nil, err
	}
	return requestBody, nil
}

func (r *OpenAIRouter) recordCompressionPlan(
	ctx *RequestContext,
	result contextcompression.ServiceResult,
) {
	ctx.ContextCompressionStrategy = "extractive"
	if len(result.RecoveryKeys) > 0 {
		ctx.ContextCompressionStrategy = "recoverable"
	}
	ctx.ContextCompressionBudgetMode = "request_and_item"
	ctx.ContextCompressionTokenSource = result.Plan.TokenCounterSource
	ctx.ContextCompressionTrigger = result.Plan.TriggerReason
	ctx.ContextCompressionQuality = result.Plan.Quality
	ctx.ContextCompressionFallback = result.Plan.FallbackReason
	status := "applied"
	if !result.Applied {
		status = string(result.Plan.SkipReason)
	}
	metrics.RecordContextCompressionPlan(
		requestDecisionStateKey(ctx),
		ctx.ContextCompressionStrategy,
		"mixed",
		status,
		ctx.ContextCompressionTokenSource,
		result.TokensBefore,
		result.TokensAfter,
		len(result.RecoveryKeys),
		result.Plan.Quality,
		result.Plan.FallbackReason,
	)
	savedTokens := max(0, result.TokensBefore-result.TokensAfter)
	r.recordContextCompressionCostSavings(ctx, savedTokens)
}

func (r *OpenAIRouter) recordContextCompressionCostSavings(
	ctx *RequestContext,
	savedTokens int,
) {
	if r == nil || r.Config == nil || savedTokens <= 0 {
		return
	}
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || params.Pricing.PromptPer1M <= 0 {
		return
	}
	ctx.ContextCompressionCostSaved = float64(savedTokens) *
		params.Pricing.PromptPer1M / 1_000_000
	metrics.RecordContextCompressionCostSavings(
		model,
		params.Pricing.Currency,
		ctx.ContextCompressionCostSaved,
	)
	if service := r.contextCompressionService(); service != nil {
		service.RecordEstimatedCostSavings(ctx.ContextCompressionCostSaved)
	}
}

func recordContextCompressionStatus(
	ctx *RequestContext,
	status string,
	latencySeconds float64,
) {
	ctx.ContextCompressionSkipReason = status
	metrics.RecordPluginExecution(
		"context_compression",
		requestDecisionStateKey(ctx),
		status,
		latencySeconds,
	)
}

func recordContextCompressionApplied(
	ctx *RequestContext,
	stats contextCompressionStats,
	start time.Time,
) {
	ctx.ContextCompressionApplied = true
	ctx.ContextCompressionBefore = stats.beforeTokens
	ctx.ContextCompressionAfter = stats.afterTokens
	ctx.ContextCompressionMessages = stats.appliedMessages
	ctx.ContextCompressionFormat = stats.format()
	ctx.ContextCompressionOmitted = stats.omittedChunks
	ctx.ContextCompressionSkipReason = ""
	metrics.RecordContextCompression(
		stats.beforeTokens,
		stats.afterTokens,
		ctx.ContextCompressionFormat,
	)
	metrics.RecordPluginExecution(
		"context_compression",
		requestDecisionStateKey(ctx),
		"applied",
		time.Since(start).Seconds(),
	)
	logging.ComponentDebugEvent("extproc", "context_compression_applied", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"messages":       stats.appliedMessages,
		"blocks":         stats.appliedBlocks,
		"tokens_before":  stats.beforeTokens,
		"tokens_after":   stats.afterTokens,
		"tokens_saved":   stats.beforeTokens - stats.afterTokens,
		"omitted_chunks": stats.omittedChunks,
		"json_blocks":    stats.jsonBlocks,
		"rag_skipped":    stats.skippedRAG,
		"latency_millis": time.Since(start).Milliseconds(),
	})
}

func isRAGToolResult(value map[string]interface{}) bool {
	for _, field := range []string{"tool_call_id", "tool_use_id"} {
		id, _ := value[field].(string)
		if strings.HasPrefix(id, "rag_") {
			return true
		}
	}
	return false
}
