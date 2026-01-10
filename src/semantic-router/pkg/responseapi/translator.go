package responseapi

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ChatCompletionRequest represents an OpenAI Chat Completions API request.
type ChatCompletionRequest struct {
	Model            string             `json:"model"`
	Messages         []ChatMessage      `json:"messages"`
	Temperature      *float64           `json:"temperature,omitempty"`
	TopP             *float64           `json:"top_p,omitempty"`
	MaxTokens        *int               `json:"max_tokens,omitempty"`
	Stream           bool               `json:"stream,omitempty"`
	Tools            []ChatTool         `json:"tools,omitempty"`
	ToolChoice       interface{}        `json:"tool_choice,omitempty"`
	ResponseFormat   interface{}        `json:"response_format,omitempty"`
	User             string             `json:"user,omitempty"`
	N                *int               `json:"n,omitempty"`
	Stop             interface{}        `json:"stop,omitempty"`
	PresencePenalty  *float64           `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64           `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64 `json:"logit_bias,omitempty"`
	Seed             *int               `json:"seed,omitempty"`
}

// ChatMessage represents a message in the Chat Completions API.
type ChatMessage struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"` // string or []ContentPart
	Name       string      `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool call in assistant messages.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call details.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatTool represents a tool in Chat Completions API.
type ChatTool struct {
	Type     string       `json:"type"`
	Function *FunctionDef `json:"function,omitempty"`
}

// ChatCompletionResponse represents an OpenAI Chat Completions API response.
type ChatCompletionResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []Choice         `json:"choices"`
	Usage   *CompletionUsage `json:"usage,omitempty"`
}

// Choice represents a choice in the response.
type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// CompletionUsage represents token usage in completions.
type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Translator handles conversion between Response API and Chat Completions API.
type Translator struct{}

// NewTranslator creates a new translator instance.
func NewTranslator() *Translator {
	return &Translator{}
}

// TranslateToCompletionRequest converts a Response API request to Chat Completions request.
func (t *Translator) TranslateToCompletionRequest(
	req *ResponseAPIRequest,
	history []*StoredResponse,
) (*ChatCompletionRequest, error) {
	messages := []ChatMessage{}

	// Add system instructions if provided, otherwise inherit from the conversation chain.
	instructions := req.Instructions
	if instructions == "" {
		for _, resp := range history {
			if resp != nil && resp.Instructions != "" {
				instructions = resp.Instructions
				break
			}
		}
	}
	if instructions != "" {
		messages = append(messages, ChatMessage{
			Role:    RoleSystem,
			Content: instructions,
		})
	}

	// Add history from previous responses
	for _, resp := range history {
		// Add input items from history
		for _, item := range resp.Input {
			msg, err := t.inputItemToMessage(item)
			if err != nil {
				continue
			}
			messages = append(messages, msg)
		}
		// Add output items from history
		for _, item := range resp.Output {
			msg, err := t.outputItemToMessage(item)
			if err != nil {
				continue
			}
			messages = append(messages, msg)
		}
	}

	// Add current input
	inputMessages, err := t.parseInput(req.Input)
	if err != nil {
		return nil, fmt.Errorf("failed to parse input: %w", err)
	}
	messages = append(messages, inputMessages...)

	// Build the request
	completionReq := &ChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxOutputTokens,
		Stream:      req.Stream,
	}

	// Convert tools
	if len(req.Tools) > 0 {
		completionReq.Tools = t.convertTools(req.Tools)
		completionReq.ToolChoice = req.ToolChoice
	}

	return completionReq, nil
}

// TranslateToResponseAPIResponse converts a Chat Completions response to Response API response.
func (t *Translator) TranslateToResponseAPIResponse(
	req *ResponseAPIRequest,
	resp *ChatCompletionResponse,
	previousResponseID string,
) *ResponseAPIResponse {
	responseID := GenerateResponseID()
	now := time.Now().Unix()

	output := []OutputItem{}
	var outputText strings.Builder

	for _, choice := range resp.Choices {
		msg := choice.Message

		if msg.Content != nil {
			// Handle text content
			contentStr, ok := msg.Content.(string)
			if ok && contentStr != "" {
				outputText.WriteString(contentStr)
				output = append(output, OutputItem{
					Type:   ItemTypeMessage,
					ID:     GenerateItemID(),
					Role:   msg.Role,
					Status: StatusCompleted,
					Content: []ContentPart{{
						Type: ContentTypeOutputText,
						Text: contentStr,
					}},
				})
			}
		}

		// Handle tool calls
		for _, tc := range msg.ToolCalls {
			output = append(output, OutputItem{
				Type:      ItemTypeFunctionCall,
				ID:        GenerateItemID(),
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				Status:    StatusCompleted,
			})
		}
	}

	var usage *Usage
	if resp.Usage != nil {
		usage = &Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
	}

	return &ResponseAPIResponse{
		ID:                 responseID,
		Object:             "response",
		CreatedAt:          now,
		Model:              resp.Model,
		Status:             StatusCompleted,
		Output:             output,
		OutputText:         outputText.String(),
		PreviousResponseID: previousResponseID,
		ConversationID:     req.ConversationID,
		Usage:              usage,
		Instructions:       req.Instructions,
		Metadata:           req.Metadata,
		Temperature:        req.Temperature,
		TopP:               req.TopP,
		MaxOutputTokens:    req.MaxOutputTokens,
		Tools:              req.Tools,
		ToolChoice:         req.ToolChoice,
	}
}

// parseInput parses the input field which can be a string or array.
func (t *Translator) parseInput(input json.RawMessage) ([]ChatMessage, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input is required")
	}

	// Try parsing as string first
	var inputStr string
	if err := json.Unmarshal(input, &inputStr); err == nil {
		return []ChatMessage{{Role: RoleUser, Content: inputStr}}, nil
	}

	// Try parsing as array of input items
	var items []InputItem
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, fmt.Errorf("invalid input format: %w", err)
	}

	messages := []ChatMessage{}
	for _, item := range items {
		msg, err := t.inputItemToMessage(item)
		if err != nil {
			continue
		}
		messages = append(messages, msg)
	}

	return messages, nil
}

// inputItemToMessage converts an InputItem to a ChatMessage.
func (t *Translator) inputItemToMessage(item InputItem) (ChatMessage, error) {
	msg := ChatMessage{Role: item.Role}
	if msg.Role == "" {
		msg.Role = RoleUser
	}

	// Parse content
	if len(item.Content) > 0 {
		var contentStr string
		if err := json.Unmarshal(item.Content, &contentStr); err == nil {
			msg.Content = contentStr
		} else {
			var parts []ContentPart
			if err := json.Unmarshal(item.Content, &parts); err == nil {
				msg.Content = t.convertContentParts(parts)
			}
		}
	}

	return msg, nil
}

// outputItemToMessage converts an OutputItem to a ChatMessage.
func (t *Translator) outputItemToMessage(item OutputItem) (ChatMessage, error) {
	switch item.Type {
	case ItemTypeMessage:
		content := ""
		for _, part := range item.Content {
			if part.Type == ContentTypeOutputText {
				content += part.Text
			}
		}
		return ChatMessage{Role: item.Role, Content: content}, nil

	case ItemTypeFunctionCall:
		return ChatMessage{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{{
				ID:   item.CallID,
				Type: "function",
				Function: FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			}},
		}, nil

	case ItemTypeFunctionCallOutput:
		return ChatMessage{
			Role:       "tool",
			Content:    item.Output,
			ToolCallID: item.CallID,
		}, nil
	}

	return ChatMessage{}, fmt.Errorf("unknown item type: %s", item.Type)
}

func (t *Translator) convertTools(tools []Tool) []ChatTool {
	result := make([]ChatTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" && tool.Function != nil {
			result = append(result, ChatTool{
				Type:     "function",
				Function: tool.Function,
			})
		}
	}
	return result
}

func (t *Translator) convertContentParts(parts []ContentPart) interface{} {
	if len(parts) == 1 && (parts[0].Type == ContentTypeInputText || parts[0].Type == ContentTypeOutputText) {
		return parts[0].Text
	}
	result := make([]map[string]interface{}, 0, len(parts))
	for _, part := range parts {
		item := map[string]interface{}{"type": part.Type}
		if part.Text != "" {
			item["text"] = part.Text
		}
		if part.ImageURL != "" {
			item["image_url"] = map[string]string{"url": part.ImageURL}
		}
		result = append(result, item)
	}
	return result
}
