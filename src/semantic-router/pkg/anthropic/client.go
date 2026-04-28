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
	for i, msg := range openAIRequest.Messages {
		switch {
		case msg.OfSystem != nil:
			systemPrompt = extractSystemContent(msg.OfSystem)
		case msg.OfUser != nil:
			content := extractUserContent(msg.OfUser)
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(content)))
		case msg.OfAssistant != nil:
			blocks := buildAssistantContentBlocks(msg.OfAssistant)
			messages = append(messages, anthropic.NewAssistantMessage(blocks...))
		case msg.OfTool != nil:
			content := extractToolContent(msg.OfTool)
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.OfTool.ToolCallID, content, false),
			))
		default:
			logging.Warnf("OpenAI msg[%d]: unrecognized message type (no OfSystem/OfUser/OfAssistant/OfTool)", i)
		}
	}

	// Log message structure before merge for debugging
	preMergeCount := len(messages)
	var preMergeRoles []string
	for _, m := range messages {
		preMergeRoles = append(preMergeRoles, fmt.Sprintf("%s(%s)", m.Role, describeContentBlocks(m.Content)))
	}
	logging.Infof("Pre-merge messages (%d): %v", preMergeCount, preMergeRoles)

	// Merge consecutive same-role messages. The OpenAI→Anthropic conversion
	// can produce adjacent user messages (e.g., a user text message followed
	// by tool result messages). Anthropic requires tool_result blocks to be
	// in the same user message immediately after the assistant's tool_use.
	messages = mergeConsecutiveSameRoleMessages(messages)

	// Validate and repair tool_use/tool_result pairing
	messages = ensureToolResultPairing(messages)

	// Reorder content blocks: tool_result must come before text in user messages
	reorderToolResultFirst(messages)

	if len(messages) != preMergeCount {
		var postFixRoles []string
		for _, m := range messages {
			postFixRoles = append(postFixRoles, fmt.Sprintf("%s(%s)", m.Role, describeContentBlocks(m.Content)))
		}
		logging.Infof("Post-fix messages (%d): %v", len(messages), postFixRoles)
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

	// Convert OpenAI tools to Anthropic format
	if len(openAIRequest.Tools) > 0 {
		var tools []anthropic.ToolUnionParam
		for _, t := range openAIRequest.Tools {
			schema := anthropic.ToolInputSchemaParam{
				Properties: map[string]any{},
			}
			if params := t.Function.Parameters; params != nil {
				if props, ok := params["properties"]; ok {
					schema.Properties = props
				}
				if req, ok := params["required"]; ok {
					if reqSlice, ok := req.([]any); ok {
						for _, r := range reqSlice {
							if s, ok := r.(string); ok {
								schema.Required = append(schema.Required, s)
							}
						}
					} else if reqStrSlice, ok := req.([]string); ok {
						schema.Required = reqStrSlice
					}
				}
			}
			tool := anthropic.ToolUnionParamOfTool(schema, t.Function.Name)
			if t.Function.Description.Value != "" {
				tool.OfTool.Description = anthropic.String(t.Function.Description.Value)
			}
			tools = append(tools, tool)
		}
		params.Tools = tools
	}

	data, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	// Patch for Vertex AI rawPredict: strip "model" (specified in URL path,
	// not body) and add "anthropic_version" (required by Vertex).
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return data, nil
	}
	delete(raw, "model")
	raw["anthropic_version"] = json.RawMessage(`"vertex-2023-10-16"`)

	// Log all top-level keys being sent to the provider for debugging
	keys := make([]string, 0, len(raw))
	for k := range raw {
		keys = append(keys, k)
	}
	logging.Infof("Anthropic request body keys: %v, body size: %d bytes", keys, len(data))

	// Dump full messages JSON for debugging
	if msgsRaw, ok := raw["messages"]; ok {
		logging.Infof("Anthropic messages JSON (%d bytes): %s", len(msgsRaw), string(msgsRaw))
	}

	return json.Marshal(raw)
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
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIToolCallFunc `json:"function"`
}

type openAIToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
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
	logging.Debugf("Raw Anthropic response (%d bytes): %.500s", len(anthropicResponse), string(anthropicResponse))

	var resp anthropic.Message
	if err := json.Unmarshal(anthropicResponse, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	// Detect provider error responses (Vertex AI returns {"type":"error",...})
	// which unmarshal into an empty Message with no ID or content.
	if resp.ID == "" && len(resp.Content) == 0 {
		logging.Warnf("Provider returned error response (%d bytes): %s", len(anthropicResponse), string(anthropicResponse))
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(anthropicResponse, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("provider error: %s", errResp.Error.Message)
		}
		return nil, fmt.Errorf("provider returned empty response (%d bytes)", len(anthropicResponse))
	}

	logging.Debugf("Parsed Anthropic response - ID: %s, Content blocks: %d, StopReason: %s", resp.ID, len(resp.Content), resp.StopReason)

	// Extract text content, thinking content, and tool calls from content blocks
	var content string
	var thinkingContent string
	var toolCalls []openAIToolCall
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "thinking":
			thinkingContent += block.Thinking
			logging.Debugf("Extracted thinking block (%d chars)", len(block.Thinking))
		case "tool_use":
			tc := block.AsToolUse()
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: openAIToolCallFunc{
					Name:      tc.Name,
					Arguments: string(tc.Input),
				},
			})
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
		logging.Debugf("Anthropic thinking response: %d chars thinking, %d chars text, estimated %d reasoning tokens",
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
				ToolCalls:        toolCalls,
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

// buildAssistantContentBlocks builds Anthropic content blocks from an OpenAI assistant message,
// including both text and tool_use blocks when tool calls are present.
func buildAssistantContentBlocks(msg *openai.ChatCompletionAssistantMessageParam) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion

	text := extractAssistantContent(msg)
	if text != "" {
		blocks = append(blocks, anthropic.NewTextBlock(text))
	}

	for _, tc := range msg.ToolCalls {
		var input any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
			input = map[string]any{}
		}
		blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, input, tc.Function.Name))
	}

	if len(blocks) == 0 {
		blocks = append(blocks, anthropic.NewTextBlock(""))
	}

	return blocks
}

// extractToolContent extracts text content from a tool result message.
func extractToolContent(msg *openai.ChatCompletionToolMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// reorderToolResultFirst ensures tool_result blocks appear before text blocks
// in user messages. Anthropic requires tool_result blocks to be the first
// content blocks in a user message that follows an assistant tool_use.
func reorderToolResultFirst(messages []anthropic.MessageParam) {
	for i := range messages {
		if messages[i].Role != "user" {
			continue
		}
		content := messages[i].Content
		hasToolResult := false
		for _, b := range content {
			if b.OfToolResult != nil {
				hasToolResult = true
				break
			}
		}
		if !hasToolResult {
			continue
		}

		// Partition: tool_result blocks first, then everything else
		var toolResults, other []anthropic.ContentBlockParamUnion
		for _, b := range content {
			if b.OfToolResult != nil {
				toolResults = append(toolResults, b)
			} else {
				other = append(other, b)
			}
		}
		messages[i].Content = append(toolResults, other...)
	}
}

// describeContentBlocks returns a compact summary of block types for logging.
func describeContentBlocks(blocks []anthropic.ContentBlockParamUnion) string {
	counts := map[string]int{}
	for _, b := range blocks {
		switch {
		case b.OfText != nil:
			counts["text"]++
		case b.OfToolUse != nil:
			counts["tool_use"]++
		case b.OfToolResult != nil:
			counts["tool_result"]++
		case b.OfThinking != nil:
			counts["thinking"]++
		default:
			counts["other"]++
		}
	}
	var parts []string
	for _, k := range []string{"text", "tool_use", "tool_result", "thinking", "other"} {
		if c := counts[k]; c > 0 {
			parts = append(parts, fmt.Sprintf("%dx%s", c, k))
		}
	}
	return strings.Join(parts, ",")
}

// mergeConsecutiveSameRoleMessages combines adjacent messages with the same role
// into a single message by appending their content blocks. This is required
// because the OpenAI format uses separate "tool" messages that become separate
// "user" messages in Anthropic format, but Anthropic requires all tool_result
// blocks to be in one user message immediately after the assistant's tool_use.
func mergeConsecutiveSameRoleMessages(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) <= 1 {
		return messages
	}
	merged := []anthropic.MessageParam{messages[0]}
	for i := 1; i < len(messages); i++ {
		last := &merged[len(merged)-1]
		if messages[i].Role == last.Role {
			last.Content = append(last.Content, messages[i].Content...)
		} else {
			merged = append(merged, messages[i])
		}
	}
	return merged
}

// ensureToolResultPairing validates and repairs tool_use/tool_result pairing.
// Anthropic requires that every tool_use block in an assistant message has a
// corresponding tool_result in the immediately following user message. After the
// OpenAI→Anthropic conversion and merge, this invariant can be broken if the
// proxy reordered messages or if content blocks ended up in the wrong position.
func ensureToolResultPairing(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) < 2 {
		return messages
	}

	repaired := false
	for i := 0; i < len(messages)-1; i++ {
		if messages[i].Role != "assistant" {
			continue
		}

		// Collect tool_use IDs from this assistant message
		var toolUseIDs []string
		for _, block := range messages[i].Content {
			if block.OfToolUse != nil {
				toolUseIDs = append(toolUseIDs, block.OfToolUse.ID)
			}
		}
		if len(toolUseIDs) == 0 {
			continue
		}

		// Check the next message — must be a user message
		if i+1 >= len(messages) || messages[i+1].Role != "user" {
			// No user message follows the assistant — insert an empty one
			logging.Warnf("ensureToolResultPairing: assistant msg[%d] has %d tool_use blocks but no user message follows; inserting one", i, len(toolUseIDs))
			emptyUser := anthropic.MessageParam{
				Role:    "user",
				Content: nil,
			}
			// Insert at i+1
			messages = append(messages[:i+1+1], messages[i+1:]...)
			messages[i+1] = emptyUser
			repaired = true
		}

		// Collect tool_result IDs already present in the next user message
		nextUser := &messages[i+1]
		presentIDs := make(map[string]bool)
		for _, block := range nextUser.Content {
			if block.OfToolResult != nil {
				presentIDs[block.OfToolResult.ToolUseID] = true
			}
		}

		// Find missing tool_results
		for _, tuID := range toolUseIDs {
			if presentIDs[tuID] {
				continue
			}

			// Search later user messages for this tool_result
			found := false
			for j := i + 2; j < len(messages); j++ {
				if messages[j].Role != "user" {
					continue
				}
				for k := 0; k < len(messages[j].Content); k++ {
					block := messages[j].Content[k]
					if block.OfToolResult != nil && block.OfToolResult.ToolUseID == tuID {
						// Move this block to the correct user message
						logging.Warnf("ensureToolResultPairing: moving tool_result(%s) from msg[%d] to msg[%d]", tuID, j, i+1)
						nextUser.Content = append(nextUser.Content, block)
						messages[j].Content = append(messages[j].Content[:k], messages[j].Content[k+1:]...)
						found = true
						repaired = true
						break
					}
				}
				if found {
					break
				}
			}

			if !found {
				logging.Errorf("ensureToolResultPairing: tool_use(%s) in msg[%d] has NO matching tool_result anywhere", tuID, i)
			}
		}
	}

	// Remove any user messages that became empty after block relocation
	if repaired {
		var cleaned []anthropic.MessageParam
		for _, m := range messages {
			if m.Role == "user" && len(m.Content) == 0 {
				continue
			}
			cleaned = append(cleaned, m)
		}
		// Re-merge in case removal created consecutive same-role messages
		return mergeConsecutiveSameRoleMessages(cleaned)
	}

	return messages
}
