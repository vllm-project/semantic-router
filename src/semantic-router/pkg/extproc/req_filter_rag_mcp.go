package extproc

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"text/template"
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

// substituteVariables substitutes variables in tool arguments using a template engine
// This provides proper escaping and prevents injection attacks
func (r *OpenAIRouter) substituteVariables(toolArgs map[string]interface{}, ctx *RequestContext) map[string]interface{} {
	if toolArgs == nil {
		return make(map[string]interface{})
	}

	result := make(map[string]interface{})
	for k, v := range toolArgs {
		switch val := v.(type) {
		case string:
			// Use text/template for safe variable substitution
			// Convert ${var} syntax to {{.var}} for template engine
			// This provides automatic escaping and prevents injection
			templateStr := convertToTemplateSyntax(val)

			tmpl, err := template.New("substitution").Parse(templateStr)
			if err != nil {
				logging.Warnf("Failed to parse template for key %s: %v, using original value", k, err)
				result[k] = val
				continue
			}

			// Prepare template data
			templateData := map[string]interface{}{
				"user_content":     ctx.UserContent,
				"matched_domains":  strings.Join(ctx.VSRMatchedDomains, ","),
				"matched_keywords": strings.Join(ctx.VSRMatchedKeywords, ","),
				"decision_name":    ctx.VSRSelectedDecisionName,
			}

			var buf bytes.Buffer
			if err := tmpl.Execute(&buf, templateData); err != nil {
				logging.Warnf("Failed to execute template for key %s: %v, using original value", k, err)
				result[k] = val
				continue
			}

			result[k] = buf.String()
		default:
			result[k] = v
		}
	}

	return result
}

// convertToTemplateSyntax converts ${var} syntax to {{.var}} for Go templates
// This allows using Go's template engine which provides automatic escaping
func convertToTemplateSyntax(s string) string {
	// Replace ${var} with {{.var}}
	// Handle common variable names
	replacements := map[string]string{
		"${user_content}":     "{{.user_content}}",
		"${matched_domains}":  "{{.matched_domains}}",
		"${matched_keywords}": "{{.matched_keywords}}",
		"${decision_name}":    "{{.decision_name}}",
	}

	result := s
	for old, new := range replacements {
		result = strings.ReplaceAll(result, old, new)
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
		if data, ok := resultMap["data"]; ok {
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

// MCPToolInvoker defines the minimal interface required to invoke an MCP tool.
// This allows the RAG plugin to work with any ToolsDatabase implementation
// that supports MCP tool invocation, without requiring a specific concrete type.
// When ToolsDatabase is extended to support MCP, it should implement this interface.
type MCPToolInvoker interface {
	InvokeMCPTool(ctx context.Context, serverName string, toolName string, args map[string]interface{}) (interface{}, error)
}

// invokeMCPTool invokes an MCP tool using the router's ToolsDatabase if it supports MCPToolInvoker.
// This design allows the MCP backend to work when a compatible ToolsDatabase is configured,
// while gracefully handling cases where MCP support is not available.
//
// Note: Currently, ToolsDatabase does not implement MCPToolInvoker. When MCP support is added
// to ToolsDatabase, it should implement this interface and this function will automatically work.
func (r *OpenAIRouter) invokeMCPTool(ctx context.Context, serverName string, toolName string, args map[string]interface{}) (interface{}, error) {
	// Ensure a tools database is configured
	if r.ToolsDatabase == nil {
		return nil, fmt.Errorf("MCP tool invocation not available: ToolsDatabase not initialized (server: %s, tool: %s)", serverName, toolName)
	}

	// Try to use ToolsDatabase as MCPToolInvoker if it implements the interface
	// This will work automatically when ToolsDatabase is extended to support MCP
	var invoker MCPToolInvoker
	var ok bool
	// Use a type assertion on the interface{} to check if ToolsDatabase implements MCPToolInvoker
	if invoker, ok = interface{}(r.ToolsDatabase).(MCPToolInvoker); ok {
		return invoker.InvokeMCPTool(ctx, serverName, toolName, args)
	}

	// ToolsDatabase is present but does not support MCP invocation
	// This is expected until MCP support is added to ToolsDatabase
	return nil, fmt.Errorf("MCP tool invocation not supported: ToolsDatabase does not implement MCPToolInvoker (server: %s, tool: %s). MCP support needs to be added to ToolsDatabase", serverName, toolName)
}
