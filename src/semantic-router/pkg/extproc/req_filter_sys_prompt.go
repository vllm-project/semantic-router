package extproc

import (
	"encoding/json"
	"time"

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

	// Prefer the runtime-owned router config when available so request-time
	// system-prompt resolution tracks reloads without routing through globals.
	globalConfig := r.currentConfig()
	var decision *config.Decision
	if globalConfig != nil {
		decision = globalConfig.GetDecisionByName(categoryName)
	}

	// If not found in global config, fall back to router's config
	if decision == nil {
		decision = r.Classifier.GetDecisionByName(categoryName)
	}

	// Get system prompt configuration from plugins
	systemPromptConfig := decision.GetSystemPromptConfig()
	if decision == nil || systemPromptConfig == nil || systemPromptConfig.SystemPrompt == "" {
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

// addSystemPromptToRequestBody adds system prompt to the JSON request body

// addSystemPromptToRequestBody adds a system prompt to the beginning of the messages array in the JSON request body
// Returns the modified body, whether the system prompt was actually injected, and any error
func addSystemPromptToRequestBody(requestBody []byte, systemPrompt string, mode string) ([]byte, bool, error) {
	if systemPrompt == "" {
		return requestBody, false, nil
	}

	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, false, err
	}

	messagesInterface, ok := requestMap["messages"]
	if !ok {
		return requestBody, false, nil
	}
	messages, ok := messagesInterface.([]interface{})
	if !ok {
		return requestBody, false, nil
	}

	existingSystemContent, hasSystemMessage := extractExistingSystemMessage(messages)
	finalSystemContent, logMessage := resolveSystemPromptInjection(systemPrompt, mode, existingSystemContent, hasSystemMessage)
	messages = upsertLeadingSystemMessage(messages, finalSystemContent, hasSystemMessage)

	logging.Infof("%s (mode: %s)", logMessage, mode)
	requestMap["messages"] = messages
	modifiedBody, err := json.Marshal(requestMap)
	return modifiedBody, true, err
}
