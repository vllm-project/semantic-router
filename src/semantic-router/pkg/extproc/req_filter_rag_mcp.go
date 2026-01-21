package extproc

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// retrieveFromMCP retrieves context using MCP tools
func (r *OpenAIRouter) retrieveFromMCP(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	mcpConfig, ok := ragConfig.BackendConfig.(*config.MCPRAGConfig)
	if !ok {
		return "", fmt.Errorf("invalid MCP RAG config")
	}

	// Build tool arguments with variable substitution
	toolArgs := r.substituteVariables(mcpConfig.ToolArguments, ctx)

	// Set timeout
	timeout := 10 * time.Second
	if mcpConfig.TimeoutSeconds != nil {
		timeout = time.Duration(*mcpConfig.TimeoutSeconds) * time.Second
	}

	// Create context with timeout
	mcpCtx, cancel := context.WithTimeout(traceCtx, timeout)
	defer cancel()

	// Invoke MCP tool using tools database
	// Note: MCP integration may need to be added to OpenAIRouter or accessed via ToolsDatabase
	// For now, we'll use a placeholder that can be implemented when MCP client is available
	start := time.Now()

	// TODO: Implement MCP tool invocation when MCP client is available
	// This requires MCP client to be initialized in OpenAIRouter or accessible via ToolsDatabase
	result, err := r.invokeMCPTool(mcpCtx, mcpConfig.ServerName, mcpConfig.ToolName, toolArgs)
	if err != nil {
		return "", fmt.Errorf("MCP tool invocation failed: %w", err)
	}

	latency := time.Since(start).Seconds()
	ctx.RAGRetrievalLatency = latency

	// Extract context from tool result
	context, err := r.extractContextFromMCPResult(result)
	if err != nil {
		return "", fmt.Errorf("failed to extract context from MCP result: %w", err)
	}

	logging.Infof("Retrieved context via MCP tool (latency: %.3fs, server: %s, tool: %s)",
		latency, mcpConfig.ServerName, mcpConfig.ToolName)
	return context, nil
}

// substituteVariables substitutes variables in tool arguments
func (r *OpenAIRouter) substituteVariables(toolArgs map[string]interface{}, ctx *RequestContext) map[string]interface{} {
	if toolArgs == nil {
		return make(map[string]interface{})
	}

	result := make(map[string]interface{})
	for k, v := range toolArgs {
		switch val := v.(type) {
		case string:
			// Substitute variables
			substituted := strings.ReplaceAll(val, "${user_content}", ctx.UserContent)
			substituted = strings.ReplaceAll(substituted, "${matched_domains}", strings.Join(ctx.VSRMatchedDomains, ","))
			substituted = strings.ReplaceAll(substituted, "${matched_keywords}", strings.Join(ctx.VSRMatchedKeywords, ","))
			substituted = strings.ReplaceAll(substituted, "${decision_name}", ctx.VSRSelectedDecisionName)
			result[k] = substituted
		default:
			result[k] = v
		}
	}

	return result
}

// extractContextFromMCPResult extracts context from MCP tool result
func (r *OpenAIRouter) extractContextFromMCPResult(result interface{}) (string, error) {
	// MCP tool results can be in various formats
	// Try to extract content from common formats

	if resultMap, ok := result.(map[string]interface{}); ok {
		// Try "content" field
		if content, ok := resultMap["content"].(string); ok {
			return content, nil
		}

		// Try "text" field
		if text, ok := resultMap["text"].(string); ok {
			return text, nil
		}

		// Try "result" field
		if resultStr, ok := resultMap["result"].(string); ok {
			return resultStr, nil
		}

		// Try "data" field
		if data, ok := resultMap["data"].(interface{}); ok {
			if dataStr, ok := data.(string); ok {
				return dataStr, nil
			}
			if dataMap, ok := data.(map[string]interface{}); ok {
				if content, ok := dataMap["content"].(string); ok {
					return content, nil
				}
			}
		}

		// Try "results" array
		if results, ok := resultMap["results"].([]interface{}); ok {
			var parts []string
			for _, res := range results {
				if resMap, ok := res.(map[string]interface{}); ok {
					if content, ok := resMap["content"].(string); ok {
						parts = append(parts, content)
					} else if text, ok := resMap["text"].(string); ok {
						parts = append(parts, text)
					}
				} else if resStr, ok := res.(string); ok {
					parts = append(parts, resStr)
				}
			}
			if len(parts) > 0 {
				return strings.Join(parts, "\n\n---\n\n"), nil
			}
		}
	}

	// If result is a string, return it directly
	if resultStr, ok := result.(string); ok {
		return resultStr, nil
	}

	return "", fmt.Errorf("unable to extract context from MCP result: unsupported format")
}

// invokeMCPTool invokes an MCP tool (placeholder for MCP integration)
// This should be implemented when MCP client is available in OpenAIRouter
func (r *OpenAIRouter) invokeMCPTool(ctx context.Context, serverName string, toolName string, args map[string]interface{}) (interface{}, error) {
	// MCP clients are created per-use from configuration
	// We need to get MCP client from ToolsDatabase or create one from config
	// For now, we'll use ToolsDatabase if available, otherwise return error

	if r.ToolsDatabase != nil {
		// Try to get MCP client from tools database
		// This is a placeholder - actual implementation depends on how MCP clients are managed
		// The ToolsDatabase may need to be extended to support MCP client retrieval
		return nil, fmt.Errorf("MCP tool invocation via ToolsDatabase not yet implemented (server: %s, tool: %s)", serverName, toolName)
	}

	return nil, fmt.Errorf("MCP tool invocation not available: ToolsDatabase not initialized (server: %s, tool: %s)", serverName, toolName)
}
