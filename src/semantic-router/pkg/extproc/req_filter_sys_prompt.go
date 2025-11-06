package extproc

import (
	"encoding/json"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// addSystemPromptIfConfigured adds category-specific system prompt if configured
func (r *OpenAIRouter) addSystemPromptIfConfigured(modifiedBody []byte, categoryName string, model string, ctx *RequestContext) ([]byte, error) {
	if categoryName == "" {
		return modifiedBody, nil
	}

	// Try to get the most up-to-date category configuration from global config first
	globalConfig := config.Get()
	var category *config.Category
	if globalConfig != nil {
		category = globalConfig.GetCategoryByName(categoryName)
	}

	// If not found in global config, fall back to router's config
	if category == nil {
		category = r.Classifier.GetCategoryByName(categoryName)
	}

	if category == nil || category.SystemPrompt == "" {
		return modifiedBody, nil
	}

	if !category.IsSystemPromptEnabled() {
		logging.Infof("System prompt disabled for category: %s", categoryName)
		return modifiedBody, nil
	}

	// Start system prompt injection span
	promptCtx, promptSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSystemPromptInjection)

	mode := category.GetSystemPromptMode()
	var injected bool
	var err error
	modifiedBody, injected, err = addSystemPromptToRequestBody(modifiedBody, category.SystemPrompt, mode)
	if err != nil {
		logging.Errorf("Error adding system prompt to request: %v", err)
		tracing.RecordError(promptSpan, err)
		metrics.RecordRequestError(model, "serialization_error")
		promptSpan.End()
		return nil, status.Errorf(codes.Internal, "error adding system prompt: %v", err)
	}

	tracing.SetSpanAttributes(promptSpan,
		attribute.Bool("system_prompt.injected", injected),
		attribute.String("system_prompt.mode", mode),
		attribute.String(tracing.AttrCategoryName, categoryName))

	if injected {
		ctx.VSRInjectedSystemPrompt = true
	}

	promptSpan.End()
	ctx.TraceContext = promptCtx

	return modifiedBody, nil
}

// addSystemPromptToRequestBody adds system prompt to the JSON request body

// addSystemPromptToRequestBody adds a system prompt to the beginning of the messages array in the JSON request body
// Returns the modified body, whether the system prompt was actually injected, and any error
func addSystemPromptToRequestBody(requestBody []byte, systemPrompt string, mode string) ([]byte, bool, error) {
	if systemPrompt == "" {
		return requestBody, false, nil
	}

	// Parse the JSON request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, false, err
	}

	// Get the messages array
	messagesInterface, ok := requestMap["messages"]
	if !ok {
		return requestBody, false, nil // No messages array, return original
	}

	messages, ok := messagesInterface.([]interface{})
	if !ok {
		return requestBody, false, nil // Messages is not an array, return original
	}

	// Check if there's already a system message at the beginning
	hasSystemMessage := false
	var existingSystemContent string
	if len(messages) > 0 {
		if firstMsg, ok := messages[0].(map[string]interface{}); ok {
			if role, ok := firstMsg["role"].(string); ok && role == "system" {
				hasSystemMessage = true
				if content, ok := firstMsg["content"].(string); ok {
					existingSystemContent = content
				}
			}
		}
	}

	// Handle different injection modes
	var finalSystemContent string
	var logMessage string

	switch mode {
	case "insert":
		if hasSystemMessage {
			// Insert mode: prepend category prompt to existing system message
			finalSystemContent = systemPrompt + "\n\n" + existingSystemContent
			logMessage = "Inserted category-specific system prompt before existing system message"
		} else {
			// No existing system message, just use the category prompt
			finalSystemContent = systemPrompt
			logMessage = "Added category-specific system prompt (insert mode, no existing system message)"
		}
	case "replace":
		fallthrough
	default:
		// Replace mode: use only the category prompt
		finalSystemContent = systemPrompt
		if hasSystemMessage {
			logMessage = "Replaced existing system message with category-specific system prompt"
		} else {
			logMessage = "Added category-specific system prompt to the beginning of messages"
		}
	}

	// Create the final system message
	systemMessage := map[string]interface{}{
		"role":    "system",
		"content": finalSystemContent,
	}

	if hasSystemMessage {
		// Update the existing system message
		messages[0] = systemMessage
	} else {
		// Prepend the system message to the beginning of the messages array
		messages = append([]interface{}{systemMessage}, messages...)
	}

	logging.Infof("%s (mode: %s)", logMessage, mode)

	// Update the messages in the request map
	requestMap["messages"] = messages

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(requestMap)
	return modifiedBody, true, err
}
