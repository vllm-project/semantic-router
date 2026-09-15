package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
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

	decision := ctx.VSRSelectedDecision
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

	if !gjson.ValidBytes(requestBody) {
		return nil, false, fmt.Errorf("request body is not valid JSON")
	}
	messagesResult := gjson.GetBytes(requestBody, "messages")
	if !messagesResult.Exists() || !messagesResult.IsArray() {
		return requestBody, false, nil
	}
	messages := messagesResult.Array()

	existingSystemContent, hasSystemMessage := firstSystemMessage(messages)
	finalSystemContent, logMessage := systemPromptContent(systemPrompt, mode, existingSystemContent, hasSystemMessage)
	updatedMessages, err := upsertSystemMessage(messages, finalSystemContent, hasSystemMessage)
	if err != nil {
		return nil, false, err
	}
	encoded, err := json.Marshal(updatedMessages)
	if err != nil {
		return nil, false, err
	}
	modifiedBody, err := sjson.SetRawBytes(requestBody, "messages", encoded)
	if err != nil {
		return nil, false, err
	}

	logging.Infof("%s (mode: %s)", logMessage, mode)
	return modifiedBody, true, err
}

func firstSystemMessage(messages []gjson.Result) (string, bool) {
	if len(messages) == 0 {
		return "", false
	}
	if messages[0].Get("role").String() != "system" {
		return "", false
	}
	return systemMessageContentText(messages[0].Get("content")), true
}

// systemMessageContentText returns the text of a system message whose content
// may be a plain string or an array of content parts (both valid OpenAI input).
// Without handling the structured form, the content was coerced to "" and the
// original system instructions were silently dropped during insert-mode merging.
func systemMessageContentText(content gjson.Result) string {
	if content.Type == gjson.String {
		return content.String()
	}
	if !content.IsArray() {
		return ""
	}
	parts := make([]string, 0, len(content.Array()))
	for _, item := range content.Array() {
		itemType := item.Get("type").String()
		if itemType != "" && itemType != "text" && itemType != "input_text" {
			continue
		}
		if text := item.Get("text").String(); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n")
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

func upsertSystemMessage(messages []gjson.Result, content string, hasSystemMessage bool) ([]json.RawMessage, error) {
	updated := make([]json.RawMessage, 0, len(messages)+1)
	if hasSystemMessage {
		first, err := sjson.SetBytes([]byte(messages[0].Raw), "content", content)
		if err != nil {
			return nil, err
		}
		updated = append(updated, json.RawMessage(first))
		messages = messages[1:]
	} else {
		systemMessage, err := json.Marshal(map[string]string{
			"role":    "system",
			"content": content,
		})
		if err != nil {
			return nil, err
		}
		updated = append(updated, json.RawMessage(systemMessage))
	}
	for _, message := range messages {
		updated = append(updated, json.RawMessage(message.Raw))
	}
	return updated, nil
}
