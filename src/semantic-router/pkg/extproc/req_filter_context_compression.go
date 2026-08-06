package extproc

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type contextCompressionStats struct {
	appliedMessages int
	beforeTokens    int
	afterTokens     int
}

func (r *OpenAIRouter) applyContextCompression(
	ctx *RequestContext,
	requestBody []byte,
) []byte {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return requestBody
	}
	pluginConfig := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	if pluginConfig == nil || !pluginConfig.Enabled {
		return requestBody
	}
	if contextCompressionBypassed(ctx, pluginConfig.BypassHeader) {
		metrics.RecordPluginExecution(
			"context_compression",
			requestDecisionStateKey(ctx),
			"bypassed",
			0,
		)
		return requestBody
	}

	start := time.Now()
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		metrics.RecordPluginExecution(
			"context_compression",
			requestDecisionStateKey(ctx),
			"parse_error_fail_open",
			time.Since(start).Seconds(),
		)
		return requestBody
	}
	messages, ok := request["messages"].([]interface{})
	if !ok {
		return requestBody
	}

	stats := compressToolMessages(
		messages,
		ctx.UserContent,
		pluginConfig.MinTokens,
		pluginConfig.TargetTokens,
	)
	if stats.appliedMessages == 0 {
		metrics.RecordPluginExecution(
			"context_compression",
			requestDecisionStateKey(ctx),
			"not_applied",
			time.Since(start).Seconds(),
		)
		return requestBody
	}

	compressedBody, err := json.Marshal(request)
	if err != nil {
		metrics.RecordPluginExecution(
			"context_compression",
			requestDecisionStateKey(ctx),
			"marshal_error_fail_open",
			time.Since(start).Seconds(),
		)
		return requestBody
	}
	ctx.ContextCompressionApplied = true
	ctx.ContextCompressionBefore = stats.beforeTokens
	ctx.ContextCompressionAfter = stats.afterTokens
	ctx.ContextCompressionMessages = stats.appliedMessages
	metrics.RecordPluginExecution(
		"context_compression",
		requestDecisionStateKey(ctx),
		"applied",
		time.Since(start).Seconds(),
	)
	logging.ComponentDebugEvent("extproc", "context_compression_applied", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"messages":       stats.appliedMessages,
		"tokens_before":  stats.beforeTokens,
		"tokens_after":   stats.afterTokens,
		"tokens_saved":   stats.beforeTokens - stats.afterTokens,
		"latency_millis": time.Since(start).Milliseconds(),
	})
	return compressedBody
}

func contextCompressionBypassed(ctx *RequestContext, bypassHeader string) bool {
	headerName := strings.TrimSpace(bypassHeader)
	return headerName != "" &&
		strings.EqualFold(strings.TrimSpace(headerValueCI(ctx, headerName)), "true")
}

func compressToolMessages(
	messages []interface{},
	query string,
	minTokens int,
	targetTokens int,
) contextCompressionStats {
	stats := contextCompressionStats{}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		if role != "tool" && role != "function" {
			continue
		}
		content, ok := message["content"].(string)
		if !ok {
			continue
		}
		result := contextcompression.CompressToolOutput(
			content,
			query,
			minTokens,
			targetTokens,
		)
		if !result.Applied {
			continue
		}
		message["content"] = result.Content
		stats.appliedMessages++
		stats.beforeTokens += result.OriginalTokens
		stats.afterTokens += result.CompressedTokens
	}
	return stats
}
