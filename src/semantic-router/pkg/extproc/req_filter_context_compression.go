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
	appliedBlocks   int
	beforeTokens    int
	afterTokens     int
	omittedChunks   int
	skippedRAG      int
	jsonBlocks      int
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
		recordContextCompressionStatus(ctx, "bypassed", 0)
		return requestBody
	}

	start := time.Now()
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		recordContextCompressionStatus(ctx, "parse_error_fail_open", time.Since(start).Seconds())
		return requestBody
	}
	messages, ok := request["messages"].([]interface{})
	if !ok {
		recordContextCompressionStatus(ctx, "unsupported_shape_fail_open", time.Since(start).Seconds())
		return requestBody
	}

	stats := compressRequestMessages(
		messages,
		compressionQuery(messages, ctx.UserContent),
		pluginConfig.MinTokens,
		pluginConfig.TargetTokens,
		pluginConfig.CompressRAG,
	)
	if stats.appliedMessages == 0 {
		status := "not_applied"
		if stats.skippedRAG > 0 {
			status = "rag_protected"
		}
		recordContextCompressionStatus(ctx, status, time.Since(start).Seconds())
		return requestBody
	}

	compressedBody, err := json.Marshal(request)
	if err != nil {
		recordContextCompressionStatus(ctx, "marshal_error_fail_open", time.Since(start).Seconds())
		return requestBody
	}
	recordContextCompressionApplied(ctx, stats, start)
	return compressedBody
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

func contextCompressionBypassed(ctx *RequestContext, bypassHeader string) bool {
	headerName := strings.TrimSpace(bypassHeader)
	return headerName != "" &&
		strings.EqualFold(strings.TrimSpace(headerValueCI(ctx, headerName)), "true")
}

func compressRequestMessages(
	messages []interface{},
	query string,
	minTokens int,
	targetTokens int,
	compressRAG bool,
) contextCompressionStats {
	stats := contextCompressionStats{}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		switch role {
		case "tool", "function":
			if isRAGToolResult(message) && !compressRAG {
				stats.skippedRAG++
				continue
			}
			result := compressMessageContent(message, query, minTokens, targetTokens)
			stats.addMessageResult(result)
		case "user":
			result := compressAnthropicToolResults(
				message,
				query,
				minTokens,
				targetTokens,
				compressRAG,
			)
			stats.addMessageResult(result)
		}
	}
	return stats
}

type messageCompressionResult struct {
	applied       bool
	appliedBlocks int
	beforeTokens  int
	afterTokens   int
	omittedChunks int
	skippedRAG    int
	jsonBlocks    int
}

func (stats *contextCompressionStats) addMessageResult(result messageCompressionResult) {
	if result.applied {
		stats.appliedMessages++
		stats.appliedBlocks += result.appliedBlocks
		stats.beforeTokens += result.beforeTokens
		stats.afterTokens += result.afterTokens
		stats.omittedChunks += result.omittedChunks
		stats.jsonBlocks += result.jsonBlocks
	}
	stats.skippedRAG += result.skippedRAG
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

func compressMessageContent(
	message map[string]interface{},
	query string,
	minTokens int,
	targetTokens int,
) messageCompressionResult {
	switch content := message["content"].(type) {
	case string:
		result := contextcompression.CompressToolOutput(content, query, minTokens, targetTokens)
		if !result.Applied {
			return messageCompressionResult{}
		}
		message["content"] = result.Content
		return compressionResultFromBlock(result)
	case []interface{}:
		return compressTextBlocks(content, query, minTokens, targetTokens)
	default:
		return messageCompressionResult{}
	}
}

func compressAnthropicToolResults(
	message map[string]interface{},
	query string,
	minTokens int,
	targetTokens int,
	compressRAG bool,
) messageCompressionResult {
	blocks, ok := message["content"].([]interface{})
	if !ok {
		return messageCompressionResult{}
	}
	total := messageCompressionResult{}
	for _, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]interface{})
		if !ok || block["type"] != "tool_result" {
			continue
		}
		if isRAGToolResult(block) && !compressRAG {
			total.skippedRAG++
			continue
		}
		result := compressMessageContent(block, query, minTokens, targetTokens)
		total.merge(result)
	}
	return total
}

func compressTextBlocks(
	blocks []interface{},
	query string,
	minTokens int,
	targetTokens int,
) messageCompressionResult {
	total := messageCompressionResult{}
	for _, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := block["type"].(string)
		if blockType != "text" {
			continue
		}
		text, ok := block["text"].(string)
		if !ok {
			continue
		}
		result := contextcompression.CompressToolOutput(text, query, minTokens, targetTokens)
		if !result.Applied {
			continue
		}
		block["text"] = result.Content
		total.merge(compressionResultFromBlock(result))
	}
	return total
}

func compressionResultFromBlock(
	result contextcompression.Result,
) messageCompressionResult {
	return messageCompressionResult{
		applied:       true,
		appliedBlocks: 1,
		beforeTokens:  result.OriginalTokens,
		afterTokens:   result.CompressedTokens,
		omittedChunks: result.OmittedChunks,
		jsonBlocks:    boolToInt(result.Format == "json"),
	}
}

func (result *messageCompressionResult) merge(other messageCompressionResult) {
	result.applied = result.applied || other.applied
	result.appliedBlocks += other.appliedBlocks
	result.beforeTokens += other.beforeTokens
	result.afterTokens += other.afterTokens
	result.omittedChunks += other.omittedChunks
	result.skippedRAG += other.skippedRAG
	result.jsonBlocks += other.jsonBlocks
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

func compressionQuery(messages []interface{}, fallback string) string {
	recent := make([]string, 0, 3)
	for index := len(messages) - 1; index >= 0 && len(recent) < 3; index-- {
		message, ok := messages[index].(map[string]interface{})
		if !ok || message["role"] != "user" {
			continue
		}
		text := userMessageText(message["content"])
		if strings.TrimSpace(text) != "" {
			recent = append(recent, text)
		}
	}
	if len(recent) == 0 {
		return fallback
	}
	for left, right := 0, len(recent)-1; left < right; left, right = left+1, right-1 {
		recent[left], recent[right] = recent[right], recent[left]
	}
	return strings.Join(recent, "\n")
}

func userMessageText(content interface{}) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, rawBlock := range typed {
			block, ok := rawBlock.(map[string]interface{})
			if !ok || block["type"] != "text" {
				continue
			}
			text, _ := block["text"].(string)
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}
