// Package anthropic provides transformation functions between OpenAI and Anthropic API formats.
// Used for Envoy-routed requests where the router transforms request/response bodies.
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxTokens is the default max tokens if not specified in request
const DefaultMaxTokens int64 = 4096

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
// messagesPath should include any base_url prefix (e.g. /Anthropic/v1/messages for AMD).
// When empty, AnthropicMessagesPath (/v1/messages) is used.
func BuildRequestHeaders(apiKey string, bodyLength int, messagesPath string) []HeaderKeyValue {
	if strings.TrimSpace(messagesPath) == "" {
		messagesPath = AnthropicMessagesPath
	}
	headers := []HeaderKeyValue{
		{Key: "anthropic-version", Value: AnthropicAPIVersion},
		{Key: "content-type", Value: "application/json"},
		{Key: "accept-encoding", Value: "identity"},
		{Key: ":path", Value: messagesPath},
		{Key: "content-length", Value: fmt.Sprintf("%d", bodyLength)},
	}
	if strings.TrimSpace(apiKey) != "" {
		headers = append([]HeaderKeyValue{{Key: "x-api-key", Value: apiKey}}, headers...)
	}
	return headers
}

// HeadersToRemove returns headers that should be removed when routing to Anthropic.
func HeadersToRemove() []string {
	return []string{"authorization", "content-length"}
}

// ToAnthropicRequestBody transforms an OpenAI-format request to Anthropic API format (JSON).
// This is used for Envoy-routed requests where the router transforms the body
// before forwarding to Anthropic via Envoy.
func ToAnthropicRequestBody(openAIRequest *openai.ChatCompletionNewParams) ([]byte, error) {
	systemPrompt, messages, err := buildAnthropicMessages(openAIRequest.Messages)
	if err != nil {
		return nil, err
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

	// Set system prompt if present
	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	if err := applyOpenAIToolsToAnthropicParams(&params, openAIRequest); err != nil {
		return nil, err
	}

	// Set optional parameters
	if openAIRequest.Temperature.Valid() {
		params.Temperature = anthropic.Float(openAIRequest.Temperature.Value)
	}
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

// ToOpenAIResponseBody transforms an Anthropic API response to OpenAI format.
// This is used for Envoy-routed requests where the router transforms the response
// after receiving it from Anthropic via Envoy.
func ToOpenAIResponseBody(anthropicResponse []byte, model string) ([]byte, error) {
	logging.Debugf("Raw Anthropic response (%d bytes): %s", len(anthropicResponse), string(anthropicResponse))

	var resp anthropic.Message
	if err := json.Unmarshal(anthropicResponse, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	logging.Debugf("Parsed Anthropic response - ID: %s, Content blocks: %d, StopReason: %s", resp.ID, len(resp.Content), resp.StopReason)

	toolCalls, content := openAIToolCallsFromAnthropicContent(resp.Content)

	// Map stop reason
	finishReason := "stop"
	switch resp.StopReason {
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	}
	if len(toolCalls) > 0 && finishReason == "stop" {
		finishReason = "tool_calls"
	}

	openAIResp := &openai.ChatCompletion{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:      "assistant",
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
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

func buildAnthropicMessages(
	messages []openai.ChatCompletionMessageParamUnion,
) (string, []anthropic.MessageParam, error) {
	var systemPrompt string
	var anthropicMessages []anthropic.MessageParam

	for _, msg := range messages {
		switch {
		case msg.OfSystem != nil:
			systemPrompt = extractSystemContent(msg.OfSystem)
		case msg.OfUser != nil:
			anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(extractUserContent(msg.OfUser)),
			))
		case msg.OfAssistant != nil:
			blocks, err := assistantContentBlocks(msg.OfAssistant)
			if err != nil {
				return "", nil, err
			}
			if len(blocks) > 0 {
				anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(blocks...))
			}
		case msg.OfTool != nil:
			toolUseID := strings.TrimSpace(msg.OfTool.ToolCallID)
			if toolUseID == "" {
				return "", nil, fmt.Errorf("tool message missing tool_call_id")
			}
			content := extractToolMessageContent(msg.OfTool)
			anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(toolUseID, content, false),
			))
		default:
			logging.Debugf("Skipping unsupported OpenAI message role in Anthropic conversion")
		}
	}

	return systemPrompt, anthropicMessages, nil
}

func extractToolMessageContent(msg *openai.ChatCompletionToolMessageParam) string {
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

func assistantContentBlocks(
	msg *openai.ChatCompletionAssistantMessageParam,
) ([]anthropic.ContentBlockParamUnion, error) {
	var blocks []anthropic.ContentBlockParamUnion

	content := extractAssistantContent(msg)
	if content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(content))
	}

	for _, toolCall := range msg.ToolCalls {
		toolUseID := strings.TrimSpace(toolCall.ID)
		if toolUseID == "" {
			return nil, fmt.Errorf("assistant tool_call missing id")
		}
		name := strings.TrimSpace(toolCall.Function.Name)
		if name == "" {
			return nil, fmt.Errorf("assistant tool_call %s missing function name", toolUseID)
		}
		input, err := parseToolCallArguments(toolCall.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("assistant tool_call %s: %w", toolUseID, err)
		}
		blocks = append(blocks, anthropic.NewToolUseBlock(toolUseID, input, name))
	}

	return blocks, nil
}

func parseToolCallArguments(arguments string) (any, error) {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return map[string]any{}, nil
	}
	var parsed any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return nil, fmt.Errorf("invalid tool arguments JSON: %w", err)
	}
	return parsed, nil
}

func applyOpenAIToolsToAnthropicParams(
	params *anthropic.MessageNewParams,
	openAIRequest *openai.ChatCompletionNewParams,
) error {
	if openAIRequest == nil {
		return nil
	}

	if len(openAIRequest.Tools) > 0 {
		tools, err := openAIToolsToAnthropic(openAIRequest.Tools)
		if err != nil {
			return err
		}
		params.Tools = tools
	}

	toolChoice, hasChoice := openAIToolChoiceToAnthropic(openAIRequest.ToolChoice)
	if hasChoice {
		params.ToolChoice = toolChoice
	}
	return nil
}

func openAIToolsToAnthropic(tools []openai.ChatCompletionToolParam) ([]anthropic.ToolUnionParam, error) {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for index, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			logging.Debugf("Skipping non-function OpenAI tool type %q at index %d", tool.Type, index)
			continue
		}
		name := strings.TrimSpace(tool.Function.Name)
		if name == "" {
			return nil, fmt.Errorf("tools[%d]: function name is required", index)
		}

		inputSchema, err := functionParametersToAnthropicSchema(tool.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("tools[%d] (%s): %w", index, name, err)
		}

		toolParam := anthropic.ToolParam{
			Name:        name,
			InputSchema: inputSchema,
			Type:        anthropic.ToolTypeCustom,
		}
		if desc := strings.TrimSpace(tool.Function.Description.Value); desc != "" {
			toolParam.Description = anthropic.String(desc)
		}
		result = append(result, anthropic.ToolUnionParam{OfTool: &toolParam})
	}
	return result, nil
}

func functionParametersToAnthropicSchema(
	parameters openai.FunctionParameters,
) (anthropic.ToolInputSchemaParam, error) {
	schema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}
	if parameters == nil {
		return schema, nil
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		return schema, fmt.Errorf("marshal function parameters: %w", err)
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return schema, fmt.Errorf("parse function parameters: %w", err)
	}
	if schema.Type == "" {
		schema.Type = "object"
	}
	return schema, nil
}

func openAIToolChoiceToAnthropic(
	choice openai.ChatCompletionToolChoiceOptionUnionParam,
) (anthropic.ToolChoiceUnionParam, bool) {
	if !param.IsOmitted(choice.OfAuto) {
		value := strings.TrimSpace(choice.OfAuto.Value)
		switch value {
		case "", "auto":
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{},
			}, true
		case "none":
			none := anthropic.NewToolChoiceNoneParam()
			return anthropic.ToolChoiceUnionParam{OfNone: &none}, true
		case "required", "any":
			return anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{},
			}, true
		default:
			logging.Debugf("Unknown OpenAI tool_choice auto value %q, defaulting to auto", value)
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{},
			}, true
		}
	}

	if choice.OfChatCompletionNamedToolChoice != nil {
		name := strings.TrimSpace(choice.OfChatCompletionNamedToolChoice.Function.Name)
		if name == "" {
			return anthropic.ToolChoiceUnionParam{}, false
		}
		return anthropic.ToolChoiceParamOfTool(name), true
	}

	return anthropic.ToolChoiceUnionParam{}, false
}

func openAIToolCallsFromAnthropicContent(
	blocks []anthropic.ContentBlockUnion,
) ([]openai.ChatCompletionMessageToolCall, string) {
	var contentParts []string
	var toolCalls []openai.ChatCompletionMessageToolCall

	for _, block := range blocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				contentParts = append(contentParts, block.Text)
			}
		case "tool_use":
			arguments := string(block.Input)
			if len(block.Input) == 0 {
				arguments = "{}"
			}
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      block.Name,
					Arguments: arguments,
				},
			})
		default:
			logging.Debugf("Skipping unsupported Anthropic content block type %q in OpenAI response conversion", block.Type)
		}
	}

	return toolCalls, strings.Join(contentParts, "")
}
