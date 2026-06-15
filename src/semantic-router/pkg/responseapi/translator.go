package responseapi

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Translator handles conversion between Response API and Chat Completions API.
type Translator struct{}

// NewTranslator creates a new translator instance.
func NewTranslator() *Translator {
	return &Translator{}
}

// resolveInstructions returns the effective system instructions, falling back to
// the most recent non-empty instructions in the conversation history.
func resolveInstructions(req *ResponseAPIRequest, history []*StoredResponse) string {
	if req.Instructions != "" {
		return req.Instructions
	}
	for _, resp := range history {
		if resp != nil && resp.Instructions != "" {
			return resp.Instructions
		}
	}
	return ""
}

// buildHistoryMessages converts stored conversation history into chat messages.
func (t *Translator) buildHistoryMessages(history []*StoredResponse) []openai.ChatCompletionMessageParamUnion {
	var msgs []openai.ChatCompletionMessageParamUnion
	for _, resp := range history {
		for _, item := range resp.Input {
			msg, err := t.inputItemToMessage(item)
			if err != nil {
				logging.Warnf("Response API: skipping malformed history input item (role=%s): %v", item.Role, err)
				continue
			}
			msgs = append(msgs, msg)
		}
		for _, item := range resp.Output {
			msg, err := t.outputItemToMessage(item)
			if err != nil {
				logging.Warnf("Response API: skipping malformed history output item (type=%s): %v", item.Type, err)
				continue
			}
			msgs = append(msgs, msg)
		}
	}
	return msgs
}

// TranslateToCompletionRequest converts a Response API request to Chat Completions request.
func (t *Translator) TranslateToCompletionRequest(
	req *ResponseAPIRequest,
	history []*StoredResponse,
) (*openai.ChatCompletionNewParams, error) {
	var messages []openai.ChatCompletionMessageParamUnion

	if instructions := resolveInstructions(req, history); instructions != "" {
		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(instructions),
				},
			},
		})
	}

	messages = append(messages, t.buildHistoryMessages(history)...)

	inputMessages, err := t.parseInput(req.Input)
	if err != nil {
		return nil, fmt.Errorf("failed to parse input: %w", err)
	}
	messages = append(messages, inputMessages...)

	completionReq := &openai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: messages,
	}

	if req.Temperature != nil {
		completionReq.Temperature = openai.Float(*req.Temperature)
	}
	if req.TopP != nil {
		completionReq.TopP = openai.Float(*req.TopP)
	}
	if req.MaxOutputTokens != nil {
		completionReq.MaxTokens = openai.Int(int64(*req.MaxOutputTokens))
	}
	if req.Stream {
		completionReq.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		}
	}

	if len(req.Tools) > 0 {
		completionReq.Tools = t.convertTools(req.Tools)
	}

	if req.ToolChoice != nil {
		completionReq.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	return completionReq, nil
}

// convertToolChoice maps the Response API tool_choice value (string or
// structured object) to the Chat Completions union parameter.
func convertToolChoice(v interface{}) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch tc := v.(type) {
	case string:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(tc),
		}
	default:
		raw, err := json.Marshal(tc)
		if err != nil {
			return openai.ChatCompletionToolChoiceOptionUnionParam{}
		}
		var named openai.ChatCompletionNamedToolChoiceParam
		if json.Unmarshal(raw, &named) == nil && named.Function.Name != "" {
			return openai.ChatCompletionToolChoiceOptionUnionParam{
				OfChatCompletionNamedToolChoice: &named,
			}
		}
		return openai.ChatCompletionToolChoiceOptionUnionParam{}
	}
}

// TranslateToResponseAPIResponse converts a Chat Completions response to Response API response.
func (t *Translator) TranslateToResponseAPIResponse(
	req *ResponseAPIRequest,
	resp *openai.ChatCompletion,
	previousResponseID string,
) *ResponseAPIResponse {
	responseID := GenerateResponseID()
	now := time.Now().Unix()

	output := []OutputItem{}
	var outputText strings.Builder

	for _, choice := range resp.Choices {
		msg := choice.Message

		if msg.Content != "" {
			outputText.WriteString(msg.Content)
			output = append(output, OutputItem{
				Type:   ItemTypeMessage,
				ID:     GenerateItemID(),
				Role:   string(msg.Role),
				Status: StatusCompleted,
				Content: []ContentPart{{
					Type: ContentTypeOutputText,
					Text: msg.Content,
				}},
			})
		}

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

	usage := &Usage{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
		TotalTokens:  int(resp.Usage.TotalTokens),
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
func (t *Translator) parseInput(input json.RawMessage) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input is required")
	}

	var inputStr string
	if err := json.Unmarshal(input, &inputStr); err == nil {
		return []openai.ChatCompletionMessageParamUnion{
			userMessage(inputStr),
		}, nil
	}

	var items []InputItem
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, fmt.Errorf("invalid input format: %w", err)
	}

	var messages []openai.ChatCompletionMessageParamUnion
	for _, item := range items {
		msg, err := t.inputItemToMessage(item)
		if err != nil {
			continue
		}
		messages = append(messages, msg)
	}

	return messages, nil
}

// inputItemToMessage converts an InputItem to an SDK message union.
// Handles both plain string content and multimodal content arrays
// (text + image_url) by building the appropriate SDK union variant.
func (t *Translator) inputItemToMessage(item InputItem) (openai.ChatCompletionMessageParamUnion, error) {
	role := item.Role
	if role == "" {
		role = RoleUser
	}

	if len(item.Content) == 0 {
		return messageForRole(role, ""), nil
	}

	var contentStr string
	if err := json.Unmarshal(item.Content, &contentStr); err == nil {
		return messageForRole(role, contentStr), nil
	}

	var parts []ContentPart
	if err := json.Unmarshal(item.Content, &parts); err == nil {
		if hasImageParts(parts) {
			if role == RoleUser {
				return userMessageWithParts(parts), nil
			}
			logging.Warnf("Response API: image content dropped for non-user role %q (OpenAI only supports images in user messages)", role)
		}
		return messageForRole(role, t.extractTextFromParts(parts)), nil
	}

	return messageForRole(role, ""), nil
}

// hasImageParts returns true if any part contains image content.
func hasImageParts(parts []ContentPart) bool {
	for _, p := range parts {
		if p.Type == ContentTypeInputImage && p.ImageURL != "" {
			return true
		}
	}
	return false
}

// userMessageWithParts builds a user message with multimodal content parts.
func userMessageWithParts(parts []ContentPart) openai.ChatCompletionMessageParamUnion {
	sdkParts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(parts))
	for _, p := range parts {
		switch {
		case (p.Type == ContentTypeInputText || p.Type == ContentTypeOutputText) && p.Text != "":
			sdkParts = append(sdkParts, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: p.Text},
			})
		case p.Type == ContentTypeInputImage && p.ImageURL != "":
			part := openai.ChatCompletionContentPartUnionParam{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: p.ImageURL,
					},
				},
			}
			if p.Detail != "" {
				part.OfImageURL.ImageURL.Detail = p.Detail
			}
			sdkParts = append(sdkParts, part)
		}
	}
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfArrayOfContentParts: sdkParts,
			},
		},
	}
}

// outputItemToMessage converts an OutputItem to an SDK message union.
func (t *Translator) outputItemToMessage(item OutputItem) (openai.ChatCompletionMessageParamUnion, error) {
	switch item.Type {
	case ItemTypeMessage:
		content := ""
		for _, part := range item.Content {
			if part.Type == ContentTypeOutputText {
				content += part.Text
			}
		}
		return messageForRole(item.Role, content), nil

	case ItemTypeFunctionCall:
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: []openai.ChatCompletionMessageToolCallParam{{
					ID:   item.CallID,
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      item.Name,
						Arguments: item.Arguments,
					},
				}},
			},
		}, nil

	case ItemTypeFunctionCallOutput:
		return openai.ChatCompletionMessageParamUnion{
			OfTool: &openai.ChatCompletionToolMessageParam{
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfString: openai.String(item.Output),
				},
				ToolCallID: item.CallID,
			},
		}, nil
	}

	return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown item type: %s", item.Type)
}

// convertTools converts Response API tools to Chat Completions tools.
// Only "function" type tools are passed through to the Chat Completions backend.
// Built-in tools like "image_generation", "code_interpreter", and "web_search" are
// handled by the router itself (via modality routing for image_generation) and are
// intentionally stripped from the translated request.
func (t *Translator) convertTools(tools []Tool) []openai.ChatCompletionToolParam {
	result := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != ToolTypeFunction || tool.Function == nil {
			continue
		}
		result = append(result, convertFunctionTool(tool.Function))
	}
	return result
}

func convertFunctionTool(fn *FunctionDef) openai.ChatCompletionToolParam {
	param := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        fn.Name,
			Description: openai.String(fn.Description),
		},
	}
	if params, ok := fn.Parameters.(map[string]interface{}); ok {
		param.Function.Parameters = openai.FunctionParameters(params)
	}
	return param
}

func (t *Translator) extractTextFromParts(parts []ContentPart) string {
	var texts []string
	for _, part := range parts {
		if part.Type == ContentTypeInputText || part.Type == ContentTypeOutputText {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, " ")
}

// userMessage builds a user message union from a string.
func userMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(content),
			},
		},
	}
}

// messageForRole builds a message union for the given role and string content.
func messageForRole(role, content string) openai.ChatCompletionMessageParamUnion {
	switch role {
	case RoleSystem:
		return openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(content),
				},
			},
		}
	case RoleAssistant:
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(content),
				},
			},
		}
	default:
		return userMessage(content)
	}
}
