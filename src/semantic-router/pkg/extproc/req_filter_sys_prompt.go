package extproc

import (
	"encoding/json"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// addSystemPromptIfConfigured adds category-specific system prompt if configured
func (r *OpenAIRouter) addSystemPromptIfConfigured(modifiedBody []byte, categoryName string, model string, ctx *RequestContext) ([]byte, error) {
	if categoryName == "" {
		return modifiedBody, nil
	}

	decision := r.decisionByName(categoryName)
	if decision == nil {
		return modifiedBody, nil
	}

	// Get system prompt configuration from plugins
	systemPromptConfig := decision.GetSystemPromptConfig()
	if systemPromptConfig == nil || systemPromptConfig.SystemPrompt == "" {
		return modifiedBody, nil
	}

	if !decision.IsSystemPromptEnabled() {
		logging.Infof("System prompt disabled for decision: %s", categoryName)
		return modifiedBody, nil
	}

	// Start system prompt plugin span
	startTime := time.Now()
	promptCtx, promptSpan := tracing.StartPluginSpan(ctx.TraceContext, "system_prompt", categoryName)

	mode := decision.GetSystemPromptMode()
	var injected bool
	var err error
	modifiedBody, injected, err = addSystemPromptToRequestBody(modifiedBody, systemPromptConfig.SystemPrompt, mode)
	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		logging.Errorf("Error adding system prompt to request: %v", err)
		tracing.RecordError(promptSpan, err)
		tracing.EndPluginSpan(promptSpan, "error", latencyMs, "injection_failed")
		metrics.RecordRequestError(model, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error adding system prompt: %v", err)
	}

	// Keep legacy attributes for backward compatibility
	tracing.SetSpanAttributes(promptSpan,
		attribute.Bool("system_prompt.injected", injected),
		attribute.String("system_prompt.mode", mode),
		attribute.String(tracing.AttrCategoryName, categoryName))

	if injected {
		ctx.VSRInjectedSystemPrompt = true
		tracing.EndPluginSpan(promptSpan, "success", latencyMs, "prompt_injected")
	} else {
		tracing.EndPluginSpan(promptSpan, "skipped", latencyMs, "no_injection_needed")
	}

	ctx.TraceContext = promptCtx

	return modifiedBody, nil
}

// addSystemPromptToRequestBody adds a system prompt to the beginning of the messages array in the JSON request body
// Returns the modified body, whether the system prompt was actually injected, and any error
func addSystemPromptToRequestBody(requestBody []byte, systemPrompt string, mode string) ([]byte, bool, error) {
	if systemPrompt == "" {
		return requestBody, false, nil
	}

	requestMap, messages, ok, err := parseRequestMessages(requestBody)
	if err != nil {
		return nil, false, err
	}
	if !ok {
		return requestBody, false, nil
	}

	existingSystemContent, hasSystemMessage := firstSystemMessage(messages)
	finalSystemContent, logMessage := systemPromptContent(systemPrompt, mode, existingSystemContent, hasSystemMessage)
	requestMap["messages"] = upsertSystemMessage(messages, finalSystemContent, hasSystemMessage)

	logging.Infof("%s (mode: %s)", logMessage, mode)

	modifiedBody, err := json.Marshal(requestMap)
	return modifiedBody, true, err
}

func parseRequestMessages(requestBody []byte) (map[string]interface{}, []interface{}, bool, error) {
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, nil, false, err
	}
	messagesInterface, ok := requestMap["messages"]
	if !ok {
		return requestMap, nil, false, nil
	}
	messages, ok := messagesInterface.([]interface{})
	return requestMap, messages, ok, nil
}

func firstSystemMessage(messages []interface{}) (string, bool) {
	if len(messages) == 0 {
		return "", false
	}
	firstMsg, ok := messages[0].(map[string]interface{})
	if !ok {
		return "", false
	}
	role, ok := firstMsg["role"].(string)
	if !ok || role != "system" {
		return "", false
	}
	return systemMessageContentText(firstMsg["content"]), true
}

// systemMessageContentText returns the text of a system message whose content
// may be a plain string or an array of content parts (both valid OpenAI input).
// Without handling the structured form, the content was coerced to "" and the
// original system instructions were silently dropped during insert-mode merging.
func systemMessageContentText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if t, ok := m["type"].(string); ok && t != "" && t != "text" && t != "input_text" {
				continue
			}
			if txt, ok := m["text"].(string); ok && txt != "" {
				parts = append(parts, txt)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func systemPromptContent(systemPrompt string, mode string, existingSystemContent string, hasSystemMessage bool) (string, string) {
	switch mode {
	case "insert":
		if hasSystemMessage {
			return systemPrompt + "\n\n" + existingSystemContent,
				"Inserted category-specific system prompt before existing system message"
		}
		return systemPrompt,
			"Added category-specific system prompt (insert mode, no existing system message)"
	case "replace":
		fallthrough
	default:
		if hasSystemMessage {
			return systemPrompt, "Replaced existing system message with category-specific system prompt"
		}
		return systemPrompt, "Added category-specific system prompt to the beginning of messages"
	}
}

func upsertSystemMessage(messages []interface{}, content string, hasSystemMessage bool) []interface{} {
	systemMessage := map[string]interface{}{
		"role":    "system",
		"content": content,
	}
	if hasSystemMessage {
		messages[0] = systemMessage
		return messages
	}
	return append([]interface{}{systemMessage}, messages...)
}
