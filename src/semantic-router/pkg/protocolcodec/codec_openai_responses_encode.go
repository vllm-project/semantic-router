package protocolcodec

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (OpenAIResponsesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	if err := validateResponsesEncodableRequest(request); err != nil {
		return nil, nil, err
	}
	diagnostics, diagnosticsErr := responsesRequestDiagnostics(request, policy)
	if diagnosticsErr != nil {
		return nil, diagnostics, diagnosticsErr
	}
	wire, err := encodeResponsesRequestWire(request)
	if err != nil {
		return nil, diagnostics, err
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

// responsesRequestDiagnostics surfaces neutral request state the Responses wire
// cannot carry. The Responses contract has its own context_management field,
// but re-emitting the Anthropic directive there would invent semantics.
func responsesRequestDiagnostics(request llmprotocol.Request, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if len(request.AnthropicContextManagement) == 0 {
		return diagnostics, nil
	}
	appendDroppedDiagnostic(
		&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIResponsesV1,
		"context_management", "Responses cannot carry the Anthropic context management directive",
	)
	return diagnostics, nil
}

func validateResponsesEncodableRequest(request llmprotocol.Request) error {
	if request.ReasoningDisplay != "" {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_reasoning_display",
			"Responses cannot represent reasoning display controls",
			nil,
		)
	}
	if request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens < 16 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_responses_max_output_tokens",
			"Responses requires an output token limit of at least 16",
			nil,
		)
	}
	return nil
}

func encodeResponsesRequestWire(request llmprotocol.Request) (responsesRequestWire, error) {
	wire := responsesRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		Store: request.Store, AutoStore: request.AutoStore, PreviousResponseID: request.PreviousResponseID,
		Truncation: request.Truncation, User: request.EndUserID,
		ParallelToolCalls: request.ParallelToolCalls, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, MaxOutputTokens: request.Sampling.MaxOutputTokens,
	}
	if request.Stream && request.StreamOptions.IncludeObfuscation != nil {
		wire.StreamOptions = &responsesStreamOptionsWire{IncludeObfuscation: request.StreamOptions.IncludeObfuscation}
	}
	if request.ConversationID != "" {
		wire.Conversation, _ = json.Marshal(request.ConversationID)
	}
	if request.ReasoningEffort != "" {
		wire.Reasoning = &responsesReasoningWire{Effort: request.ReasoningEffort}
	}
	items, err := encodeResponsesRequestItems(request)
	if err != nil {
		return responsesRequestWire{}, err
	}
	wire.Input, _ = json.Marshal(items)
	wire.Tools = encodeResponsesTools(request.Tools, request.ImageGeneration)
	wire.ToolChoice = encodeResponsesToolChoice(request.ToolChoice)
	wire.Text = encodeResponsesOutputFormat(request.OutputFormat)
	return wire, nil
}

func encodeResponsesRequestItems(request llmprotocol.Request) ([]responsesItemWire, error) {
	items := make([]responsesItemWire, 0, len(request.Messages))
	for _, instruction := range request.Instructions {
		encoded, err := encodeResponsesMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content}, "input")
		if err != nil {
			return nil, err
		}
		items = append(items, encoded...)
	}
	for _, message := range request.Messages {
		encoded, err := encodeResponsesMessage(message, "input")
		if err != nil {
			return nil, err
		}
		items = append(items, encoded...)
	}
	return items, nil
}

func encodeResponsesTools(input []llmprotocol.Tool, imageGeneration *llmprotocol.ImageGenerationOptions) json.RawMessage {
	if len(input) == 0 && imageGeneration == nil {
		return nil
	}
	tools := make([]responsesToolWire, 0, len(input)+1)
	for _, tool := range input {
		tools = append(tools, responsesToolWire{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict})
	}
	if imageGeneration != nil {
		tool := responsesToolWire{
			Type: "image_generation", Model: imageGeneration.Model,
			Quality: imageGeneration.Quality, Size: imageGeneration.Size,
			OutputFormat:      imageGeneration.OutputFormat,
			OutputCompression: imageGeneration.OutputCompression,
			Moderation:        imageGeneration.Moderation, Background: imageGeneration.Background,
			InputFidelity: imageGeneration.InputFidelity, PartialImages: imageGeneration.PartialImages,
			Action: imageGeneration.Action,
		}
		if imageGeneration.InputImageMask != nil {
			tool.InputImageMask = &responsesImageGenMaskWire{
				ImageURL: imageGeneration.InputImageMask.EncodedImage,
				FileID:   imageGeneration.InputImageMask.FileID,
			}
		}
		tools = append(tools, tool)
	}
	encoded, _ := json.Marshal(tools)
	return encoded
}

func encodeResponsesOutputFormat(output llmprotocol.OutputFormat) *responsesTextWire {
	if output.Kind == "" || output.Kind == llmprotocol.OutputText {
		return nil
	}
	return &responsesTextWire{Format: responsesFormatWire{
		Type: string(output.Kind), Name: output.Name,
		Description: output.Description, Strict: output.Strict, Schema: output.Schema,
	}}
}

func encodeResponsesMessage(message llmprotocol.Message, textDirection string) ([]responsesItemWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return nil, err
	}
	state := responsesMessageEncodingState{messageID: message.ID, role: role, textDirection: textDirection}
	for _, content := range message.Content {
		if err := state.appendContent(content); err != nil {
			return nil, err
		}
	}
	if err := state.flushOrdinary(); err != nil {
		return nil, err
	}
	if err := state.flushReasoning(); err != nil {
		return nil, err
	}
	return state.items, nil
}

type responsesMessageEncodingState struct {
	messageID     string
	role          string
	textDirection string
	ordinary      []llmprotocol.Content
	reasoning     []llmprotocol.Content
	items         []responsesItemWire
}

func (state *responsesMessageEncodingState) appendContent(content llmprotocol.Content) error {
	switch content.Kind {
	case llmprotocol.ContentToolCall:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendToolCall(content.ToolCall)
	case llmprotocol.ContentToolResult:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendToolResult(content.ToolResult)
	case llmprotocol.ContentReasoning:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		state.reasoning = append(state.reasoning, content)
	case llmprotocol.ContentGeneratedImage:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendGeneratedImage(content.GeneratedImage)
	default:
		if err := state.flushReasoning(); err != nil {
			return err
		}
		state.ordinary = append(state.ordinary, content)
	}
	return nil
}

func (state *responsesMessageEncodingState) flushPending() error {
	if err := state.flushOrdinary(); err != nil {
		return err
	}
	return state.flushReasoning()
}

func (state *responsesMessageEncodingState) appendGeneratedImage(image *llmprotocol.GeneratedImage) error {
	if image == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_generated_image", "generated image is invalid", nil)
	}
	item := responsesItemWire{
		Type:   "image_generation_call",
		ID:     responsesItemID(state.messageID, len(state.items), "image_generation_call"),
		Status: string(image.Status),
	}
	if image.Result != nil {
		result := *image.Result
		item.Result = &result
	}
	state.items = append(state.items, item)
	return nil
}

func (state *responsesMessageEncodingState) flushOrdinary() error {
	if len(state.ordinary) == 0 {
		return nil
	}
	content, err := encodeResponsesContent(state.ordinary, state.textDirection)
	if err != nil {
		return err
	}
	item := responsesItemWire{
		Type: "message", ID: responsesItemID(state.messageID, len(state.items), "message"),
		Role: state.role, Content: content,
	}
	if state.textDirection == "output" {
		item.Status = "completed"
	}
	state.items = append(state.items, item)
	state.ordinary = nil
	return nil
}

func (state *responsesMessageEncodingState) appendToolCall(call *llmprotocol.ToolCall) error {
	if call == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call is invalid", nil)
	}
	state.items = append(state.items, responsesItemWire{
		Type: "function_call", ID: responsesItemID(state.messageID, len(state.items), "function_call"),
		CallID: call.ID, Name: call.Name, Arguments: call.Arguments,
	})
	return nil
}

func (state *responsesMessageEncodingState) appendToolResult(result *llmprotocol.ToolResult) error {
	if result == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result is invalid", nil)
	}
	output, err := encodeResponsesContent(result.Content, "input")
	if err != nil {
		return err
	}
	state.items = append(state.items, responsesItemWire{
		Type: "function_call_output", ID: responsesItemID(state.messageID, len(state.items), "function_call_output"),
		CallID: result.CallID, Output: output,
	})
	return nil
}

func (state *responsesMessageEncodingState) flushReasoning() error {
	if len(state.reasoning) == 0 {
		return nil
	}
	item := responsesItemWire{
		Type: "reasoning", ID: responsesItemID(state.messageID, len(state.items), "reasoning"),
	}
	summaries := make([]map[string]string, 0, len(state.reasoning))
	texts := make([]map[string]string, 0, len(state.reasoning))
	for _, content := range state.reasoning {
		if content.Reasoning == llmprotocol.ReasoningScopeSummary {
			summaries = append(summaries, map[string]string{"type": "summary_text", "text": content.Text})
		} else {
			texts = append(texts, map[string]string{"type": "reasoning_text", "text": content.Text})
		}
	}
	var err error
	item.Summary, err = json.Marshal(summaries)
	if err != nil {
		return err
	}
	if len(texts) > 0 {
		item.Content, err = json.Marshal(texts)
		if err != nil {
			return err
		}
	}
	state.items = append(state.items, item)
	state.reasoning = nil
	return nil
}

func responsesItemID(messageID string, index int, kind string) string {
	if index == 0 && messageID != "" {
		return messageID
	}
	return llmprotocol.StableID("responses-item", messageID, fmt.Sprint(index), kind)
}

func decodeResponsesReasoning(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	providerOutput bool,
) ([]llmprotocol.Content, error) {
	var summaries []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	var err error
	if providerOutput {
		err = decodeProviderValue(raw, &summaries, policy)
	} else {
		err = decodeWireValue(raw, &summaries, policy)
	}
	if err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(summaries))
	for _, summary := range summaries {
		if summary.Type != "summary_text" {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_reasoning_summary", "Responses reasoning summary is unsupported", nil)
		}
		contents = append(contents, llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: summary.Text, Reasoning: llmprotocol.ReasoningScopeSummary,
		})
	}
	return contents, nil
}

func encodeResponsesContent(contents []llmprotocol.Content, direction string) (json.RawMessage, error) {
	parts := make([]responsesContentWire, 0, len(contents))
	for _, content := range contents {
		switch content.Kind {
		case llmprotocol.ContentText:
			part := responsesContentWire{Type: direction + "_text", Text: content.Text}
			if direction == "output" {
				part.Annotations = responsesAnnotations(content.Citations)
			}
			parts = append(parts, part)
		case llmprotocol.ContentRefusal:
			parts = append(parts, responsesContentWire{Type: "refusal", Refusal: content.Text})
		case llmprotocol.ContentReasoning:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "reasoning_content_position", "Responses reasoning must be encoded as an ordered reasoning item", nil)
		case llmprotocol.ContentImage:
			if direction != "input" {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "image_content_position", "Responses output messages cannot contain input images", nil)
			}
			imageURL := content.URL
			if content.Data != "" {
				imageURL = "data:" + content.MediaType + ";base64," + content.Data
			}
			parts = append(parts, responsesContentWire{Type: "input_image", ImageURL: imageURL, FileID: content.FileID, Detail: content.Detail})
		case llmprotocol.ContentFile:
			if direction != "input" {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_content_position", "Responses output messages cannot contain input files", nil)
			}
			parts = append(parts, responsesContentWire{Type: "input_file", FileURL: content.URL, FileID: content.FileID, FileData: content.Data, Filename: content.Filename, Detail: content.Detail})
		default:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as Responses", nil)
		}
	}
	return json.Marshal(parts)
}

func encodeResponsesToolChoice(choice llmprotocol.ToolChoice) json.RawMessage {
	switch choice.Mode {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
		body, _ := json.Marshal(choice.Mode)
		return body
	case llmprotocol.ToolChoiceNamed:
		body, _ := json.Marshal(map[string]string{"type": "function", "name": choice.Name})
		return body
	case llmprotocol.ToolChoiceImageGeneration:
		body, _ := json.Marshal(map[string]string{"type": "image_generation"})
		return body
	default:
		return nil
	}
}
