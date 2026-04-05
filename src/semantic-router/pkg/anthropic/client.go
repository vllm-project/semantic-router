// Package anthropic provides transformation functions between OpenAI and Anthropic API formats.
// Used for Envoy-routed requests where the router transforms request/response bodies.
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxTokens is the default max tokens if not specified in request
const DefaultMaxTokens int64 = 4096

// DefaultThinkingBudgetTokens is the default budget for thinking when enabled
const DefaultThinkingBudgetTokens int64 = 10000

// AnthropicAPIVersion is the version header required for Anthropic API
const AnthropicAPIVersion = "2023-06-01"

// AnthropicMessagesPath is the path for Anthropic messages API
const AnthropicMessagesPath = "/v1/messages"

// HeaderKeyValue represents a header key-value pair
type HeaderKeyValue struct {
	Key   string
	Value string
}

// BuildRequestHeaders returns the headers needed for an Anthropic API request.
// Returns headers as simple key-value pairs to avoid coupling to Envoy types.
func BuildRequestHeaders(apiKey string, bodyLength int) []HeaderKeyValue {
	return []HeaderKeyValue{
		{Key: "x-api-key", Value: apiKey},
		{Key: "anthropic-version", Value: AnthropicAPIVersion},
		{Key: "content-type", Value: "application/json"},
		{Key: "accept-encoding", Value: "identity"},
		{Key: ":path", Value: AnthropicMessagesPath},
		{Key: "content-length", Value: fmt.Sprintf("%d", bodyLength)},
	}
}

// HeadersToRemove returns headers that should be removed when routing to Anthropic.
func HeadersToRemove() []string {
	return []string{"authorization", "content-length"}
}

// ThinkingConfig holds thinking/reasoning configuration for Anthropic requests.
type ThinkingConfig struct {
	Enabled      bool
	BudgetTokens int64 // 0 means use DefaultThinkingBudgetTokens
}

// ToAnthropicRequestBody transforms an OpenAI-format request to Anthropic API format (JSON).
// This is used for Envoy-routed requests where the router transforms the body
// before forwarding to Anthropic via Envoy.
func ToAnthropicRequestBody(openAIRequest *openai.ChatCompletionNewParams) ([]byte, error) {
	return ToAnthropicRequestBodyWithThinking(openAIRequest, nil)
}

// ToAnthropicRequestBodyWithThinking transforms an OpenAI-format request to Anthropic API format
// with optional extended thinking support.
func ToAnthropicRequestBodyWithThinking(openAIRequest *openai.ChatCompletionNewParams, thinking *ThinkingConfig) ([]byte, error) {
	var messages []anthropic.MessageParam
	var systemPrompt string

	// Process messages - extract system prompt separately (Anthropic requirement)
	for _, msg := range openAIRequest.Messages {
		switch {
		case msg.OfSystem != nil:
			systemPrompt = extractSystemContent(msg.OfSystem)
		case msg.OfUser != nil:
			content := extractUserContent(msg.OfUser)
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(content)))
		case msg.OfAssistant != nil:
			content := extractAssistantContent(msg.OfAssistant)
			messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(content)))
		}
	}

	// Determine max tokens (required for Anthropic)
	maxTokens := DefaultMaxTokens
	if openAIRequest.MaxCompletionTokens.Value > 0 {
		maxTokens = openAIRequest.MaxCompletionTokens.Value
	} else if openAIRequest.MaxTokens.Value > 0 {
		maxTokens = openAIRequest.MaxTokens.Value
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(openAIRequest.Model),
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	// Enable extended thinking if configured
	if thinking != nil && thinking.Enabled {
		budgetTokens := thinking.BudgetTokens
		if budgetTokens <= 0 {
			budgetTokens = DefaultThinkingBudgetTokens
		}
		// max_tokens must be greater than budget_tokens (Anthropic requirement)
		if maxTokens <= budgetTokens {
			maxTokens = budgetTokens + 4096
			params.MaxTokens = maxTokens
		}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budgetTokens)
		// Temperature must not be set when thinking is enabled (Anthropic requirement)
		logging.Infof("Anthropic thinking enabled with budget_tokens=%d, max_tokens=%d", budgetTokens, maxTokens)
	} else {
		// Set optional parameters (only when thinking is disabled)
		if openAIRequest.Temperature.Valid() {
			params.Temperature = anthropic.Float(openAIRequest.Temperature.Value)
		}
	}

	// Set system prompt if present
	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	// Set optional parameters
	if openAIRequest.TopP.Valid() {
		params.TopP = anthropic.Float(openAIRequest.TopP.Value)
	}
	if len(openAIRequest.Stop.OfStringArray) > 0 {
		params.StopSequences = openAIRequest.Stop.OfStringArray
	} else if openAIRequest.Stop.OfString.Value != "" {
		params.StopSequences = []string{openAIRequest.Stop.OfString.Value}
	}

	return json.Marshal(params)
}

// openAIResponseWithThinking extends the OpenAI response with thinking content.
// We use a custom struct to include the reasoning_content field which is not part
// of the standard OpenAI SDK but is used by clients that support thinking/reasoning.
type openAIResponseWithThinking struct {
	ID                string                        `json:"id"`
	Object            string                        `json:"object"`
	Created           int64                         `json:"created"`
	Model             string                        `json:"model"`
	Choices           []openAIChoiceWithThinking     `json:"choices"`
	Usage             openAIUsageWithThinking        `json:"usage"`
}

type openAIChoiceWithThinking struct {
	Index            int                            `json:"index"`
	Message          openAIMessageWithThinking      `json:"message"`
	FinishReason     string                         `json:"finish_reason"`
}

type openAIMessageWithThinking struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type openAIUsageWithThinking struct {
	PromptTokens     int64                          `json:"prompt_tokens"`
	CompletionTokens int64                          `json:"completion_tokens"`
	TotalTokens      int64                          `json:"total_tokens"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type completionTokensDetails struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

// ToOpenAIResponseBody transforms an Anthropic API response to OpenAI format.
// This is used for Envoy-routed requests where the router transforms the response
// after receiving it from Anthropic via Envoy.
func ToOpenAIResponseBody(anthropicResponse []byte, model string) ([]byte, error) {
	logging.Infof("Raw Anthropic response (%d bytes): %.500s", len(anthropicResponse), string(anthropicResponse))

	var resp anthropic.Message
	if err := json.Unmarshal(anthropicResponse, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	logging.Debugf("Parsed Anthropic response - ID: %s, Content blocks: %d, StopReason: %s", resp.ID, len(resp.Content), resp.StopReason)

	// Extract text content and thinking content from content blocks
	var content string
	var thinkingContent string
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "thinking":
			thinkingContent += block.Thinking
			logging.Infof("Extracted thinking block (%d chars)", len(block.Thinking))
		}
	}

	// Map stop reason
	finishReason := "stop"
	switch resp.StopReason {
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	}

	// Calculate reasoning tokens from cache_creation metadata if available
	// The Anthropic API includes thinking token usage in the response usage
	var reasoningTokens int64
	if thinkingContent != "" {
		// Estimate reasoning tokens: Anthropic doesn't report them separately in
		// non-streaming mode, but we can infer from the total output vs text length.
		// For now, report the output tokens as including reasoning.
		reasoningTokens = int64(len(thinkingContent) / 4) // rough estimate
		logging.Infof("Anthropic thinking response: %d chars thinking, %d chars text, estimated %d reasoning tokens",
			len(thinkingContent), len(content), reasoningTokens)
	}

	openAIResp := &openAIResponseWithThinking{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIChoiceWithThinking{{
			Index: 0,
			Message: openAIMessageWithThinking{
				Role:             "assistant",
				Content:          content,
				ReasoningContent: thinkingContent,
			},
			FinishReason: finishReason,
		}},
		Usage: openAIUsageWithThinking{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}

	if reasoningTokens > 0 {
		openAIResp.Usage.CompletionTokensDetails = &completionTokensDetails{
			ReasoningTokens: reasoningTokens,
		}
	}

	return json.Marshal(openAIResp)
}

// extractSystemContent extracts text from a system message
func extractSystemContent(msg *openai.ChatCompletionSystemMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}

// extractUserContent extracts text from a user message
func extractUserContent(msg *openai.ChatCompletionUserMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

// extractAssistantContent extracts text from an assistant message
func extractAssistantContent(msg *openai.ChatCompletionAssistantMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}
