package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type OpenAIChatCodec struct{}

func (OpenAIChatCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIChatV1 }
func (OpenAIChatCodec) Stateless() bool                { return true }
func (OpenAIChatCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityAudioInput,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStructuredJSON, llmprotocol.CapabilityStrictJSONSchema, llmprotocol.CapabilityStrictToolSchema,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
		llmprotocol.CapabilityMultipleCandidates,
	)
}

type chatRequestWire struct {
	Model               string                 `json:"model"`
	Messages            []chatMessageWire      `json:"messages"`
	Tools               []chatToolWire         `json:"tools,omitempty"`
	ToolChoice          json.RawMessage        `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool                  `json:"parallel_tool_calls,omitempty"`
	CandidateCount      *int64                 `json:"n,omitempty"`
	Temperature         *float64               `json:"temperature,omitempty"`
	TopP                *float64               `json:"top_p,omitempty"`
	MaxTokens           *int64                 `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int64                 `json:"max_completion_tokens,omitempty"`
	Seed                *int64                 `json:"seed,omitempty"`
	FrequencyPenalty    *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64               `json:"presence_penalty,omitempty"`
	Stop                json.RawMessage        `json:"stop,omitempty"`
	ResponseFormat      *chatOutputWire        `json:"response_format,omitempty"`
	ReasoningEffort     string                 `json:"reasoning_effort,omitempty"`
	ReasoningBudget     *int64                 `json:"reasoning_budget_tokens,omitempty"`
	Stream              bool                   `json:"stream,omitempty"`
	StreamOptions       *chatStreamOptionsWire `json:"stream_options,omitempty"`
	Metadata            map[string]string      `json:"metadata,omitempty"`
}

type chatStreamOptionsWire struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type chatMessageWire struct {
	ID                 string               `json:"id,omitempty"`
	Role               string               `json:"role"`
	Content            json.RawMessage      `json:"content,omitempty"`
	Refusal            string               `json:"refusal,omitempty"`
	Reasoning          string               `json:"reasoning_content,omitempty"`
	AlternateReasoning string               `json:"reasoning,omitempty"`
	Audio              *chatAudioOutputWire `json:"audio,omitempty"`
	LegacyFunctionCall *chatLegacyCallWire  `json:"function_call,omitempty"`
	ToolCalls          []chatToolCallWire   `json:"tool_calls,omitempty"`
	ToolCallID         string               `json:"tool_call_id,omitempty"`
	Annotations        []chatAnnotationWire `json:"annotations,omitempty"`
}

type chatAudioOutputWire struct {
	ID         string `json:"id"`
	Data       string `json:"data"`
	ExpiresAt  int64  `json:"expires_at"`
	Transcript string `json:"transcript"`
}

type chatLegacyCallWire struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatAnnotationWire struct {
	Type        string                         `json:"type"`
	URLCitation *chatURLCitationAnnotationWire `json:"url_citation,omitempty"`
}

type chatURLCitationAnnotationWire struct {
	URL        string `json:"url"`
	Title      string `json:"title,omitempty"`
	StartIndex int64  `json:"start_index"`
	EndIndex   int64  `json:"end_index"`
}

type chatContentWire struct {
	Type       string              `json:"type"`
	Text       string              `json:"text,omitempty"`
	Refusal    string              `json:"refusal,omitempty"`
	ImageURL   *chatImageURLWire   `json:"image_url,omitempty"`
	InputAudio *chatInputAudioWire `json:"input_audio,omitempty"`
}

type chatImageURLWire struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatInputAudioWire struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type chatToolCallWire struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionWire `json:"function"`
}

type chatFunctionWire struct {
	Name        string          `json:"name"`
	Arguments   string          `json:"arguments,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type chatToolWire struct {
	Type     string           `json:"type"`
	Function chatFunctionWire `json:"function"`
}

type chatOutputWire struct {
	Type       string          `json:"type"`
	JSONObject json.RawMessage `json:"json_schema,omitempty"`
}

func (OpenAIChatCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire chatRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if len(wire.Messages) == 0 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"messages_required",
			"messages must contain at least one item",
			nil,
		)
	}
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		ParallelToolCalls: wire.ParallelToolCalls, CandidateCount: wire.CandidateCount,
		ReasoningEffort: wire.ReasoningEffort, ReasoningBudgetTokens: wire.ReasoningBudget,
		Sampling: llmprotocol.Sampling{
			Temperature: wire.Temperature, TopP: wire.TopP, Seed: wire.Seed,
			FrequencyPenalty: wire.FrequencyPenalty, PresencePenalty: wire.PresencePenalty,
		},
		Trusted: llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIChatV1},
	}
	if wire.MaxCompletionTokens != nil {
		request.Sampling.MaxOutputTokens = wire.MaxCompletionTokens
	} else {
		request.Sampling.MaxOutputTokens = wire.MaxTokens
	}
	stop, decodeRequestErr := decodeStop(wire.Stop)
	if decodeRequestErr != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, decodeRequestErr
	}
	request.Sampling.Stop = stop
	for index, messageWire := range wire.Messages {
		message, err := decodeChatMessage(messageWire, index, policy)
		if err != nil {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
		}
		if message.Role == llmprotocol.RoleSystem || message.Role == llmprotocol.RoleDeveloper {
			request.Instructions = append(request.Instructions, llmprotocol.InstructionBlock{Role: message.Role, Content: message.Content})
		} else {
			request.Messages = append(request.Messages, message)
		}
	}
	for _, toolWire := range wire.Tools {
		if toolWire.Type != "function" || strings.TrimSpace(toolWire.Function.Name) == "" {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools are supported", nil)
		}
		schema := toolWire.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{
			Name: toolWire.Function.Name, Description: toolWire.Function.Description,
			Strict: toolWire.Function.Strict, InputSchema: append(json.RawMessage(nil), schema...),
		})
	}
	request.ToolChoice, decodeRequestErr = decodeChatToolChoice(wire.ToolChoice)
	if decodeRequestErr != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, decodeRequestErr
	}
	if wire.ResponseFormat != nil {
		request.OutputFormat, decodeRequestErr = decodeChatOutputFormat(*wire.ResponseFormat)
		if decodeRequestErr != nil {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, decodeRequestErr
		}
	}
	return request, requestEnvelope(llmprotocol.OpenAIChatV1, body, request.Generation, policy), nil, nil
}

func decodeChatMessage(wire chatMessageWire, index int, policy llmprotocol.Policy) (llmprotocol.Message, error) {
	if wire.Audio != nil {
		return llmprotocol.Message{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_output_audio", "Chat output audio is unsupported", nil,
		)
	}
	if wire.LegacyFunctionCall != nil {
		return llmprotocol.Message{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_legacy_function_call", "legacy Chat function calls are unsupported", nil,
		)
	}
	role, err := canonicalRole(wire.Role)
	if err != nil {
		return llmprotocol.Message{}, err
	}
	message := llmprotocol.Message{ID: wire.ID, Role: role}
	if len(wire.Content) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Content), []byte("null")) {
		var text string
		if json.Unmarshal(wire.Content, &text) == nil {
			message.Content = append(message.Content, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: text})
		} else {
			var parts []chatContentWire
			if err := decodeWire(wire.Content, &parts, policy); err != nil {
				return llmprotocol.Message{}, err
			}
			for _, part := range parts {
				content, err := decodeChatContent(part)
				if err != nil {
					return llmprotocol.Message{}, err
				}
				message.Content = append(message.Content, content)
			}
		}
	}
	if wire.Refusal != "" {
		message.Content = append(message.Content, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: wire.Refusal})
	}
	reasoning := wire.Reasoning
	if reasoning == "" {
		reasoning = wire.AlternateReasoning
	}
	if reasoning != "" {
		message.Content = append(message.Content, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: reasoning})
	}
	for toolIndex, call := range wire.ToolCalls {
		id := call.ID
		if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			id = llmprotocol.StableID("chat", fmt.Sprint(index), fmt.Sprint(toolIndex), call.Function.Name, call.Function.Arguments)
		}
		if id == "" {
			return llmprotocol.Message{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "tool_call_id_required", "tool call ID is required", nil)
		}
		message.Content = append(message.Content, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: call.Function.Name, Arguments: call.Function.Arguments}})
	}
	if len(wire.Annotations) > 0 {
		citations, err := decodeChatAnnotations(wire.Annotations)
		if err != nil {
			return llmprotocol.Message{}, err
		}
		attached := false
		for contentIndex := range message.Content {
			if message.Content[contentIndex].Kind == llmprotocol.ContentText {
				message.Content[contentIndex].Citations = citations
				attached = true
				break
			}
		}
		if !attached {
			return llmprotocol.Message{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "citation_text_required", "URL citations require text content", nil)
		}
	}
	if role == llmprotocol.RoleTool {
		result := llmprotocol.ToolResult{CallID: wire.ToolCallID, Content: append([]llmprotocol.Content(nil), message.Content...)}
		message.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &result}}
	}
	return message, nil
}

func decodeChatAnnotations(wire []chatAnnotationWire) ([]llmprotocol.Citation, error) {
	citations := make([]llmprotocol.Citation, 0, len(wire))
	for _, annotation := range wire {
		if annotation.Type != "url_citation" || annotation.URLCitation == nil {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_annotation", "Chat Completions annotation is unsupported", nil)
		}
		citations = append(citations, llmprotocol.Citation{
			URL: annotation.URLCitation.URL, Title: annotation.URLCitation.Title,
			StartIndex: annotation.URLCitation.StartIndex, EndIndex: annotation.URLCitation.EndIndex,
		})
	}
	return citations, nil
}

func encodeChatAnnotations(citations []llmprotocol.Citation) []chatAnnotationWire {
	annotations := make([]chatAnnotationWire, 0, len(citations))
	for _, citation := range citations {
		annotations = append(annotations, chatAnnotationWire{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
			URL: citation.URL, Title: citation.Title, StartIndex: citation.StartIndex, EndIndex: citation.EndIndex,
		}})
	}
	return annotations
}

func decodeChatContent(part chatContentWire) (llmprotocol.Content, error) {
	switch part.Type {
	case "text", "input_text", "output_text":
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text}, nil
	case "refusal":
		text := part.Refusal
		if text == "" {
			text = part.Text
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: text}, nil
	case "image_url":
		if part.ImageURL == nil || part.ImageURL.URL == "" {
			return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_url_required", "image URL is required", nil)
		}
		if mediaType, data, inline := decodeDataURL(part.ImageURL.URL); inline {
			return llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: mediaType, Data: data, Detail: part.ImageURL.Detail}, nil
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: part.ImageURL.URL, Detail: part.ImageURL.Detail}, nil
	case "input_audio":
		if part.InputAudio == nil {
			return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "audio_required", "input audio is required", nil)
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentAudio, Data: part.InputAudio.Data, MediaType: part.InputAudio.Format}, nil
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "chat content type is unsupported", nil)
	}
}

func decodeStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_stop", "stop must be a string or string array", nil)
	}
	return many, nil
}

func decodeChatToolChoice(raw json.RawMessage) (llmprotocol.ToolChoice, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return llmprotocol.ToolChoice{}, nil
	}
	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		switch mode {
		case "auto":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}, nil
		case "none":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}, nil
		case "required":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceRequired}, nil
		}
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &named) != nil || named.Type != "function" || named.Function.Name == "" {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
	}
	return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: named.Function.Name}, nil
}

func decodeChatOutputFormat(wire chatOutputWire) (llmprotocol.OutputFormat, error) {
	switch wire.Type {
	case "", "text":
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputText}, nil
	case "json_object":
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONObject}, nil
	case "json_schema":
		var schema struct {
			Name        string          `json:"name"`
			Description string          `json:"description,omitempty"`
			Strict      *bool           `json:"strict,omitempty"`
			Schema      json.RawMessage `json:"schema"`
		}
		if json.Unmarshal(wire.JSONObject, &schema) != nil || len(schema.Schema) == 0 {
			return llmprotocol.OutputFormat{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json_schema", "response JSON Schema is invalid", nil)
		}
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONSchema, Name: schema.Name, Description: schema.Description, Strict: schema.Strict, Schema: schema.Schema}, nil
	default:
		return llmprotocol.OutputFormat{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "response format is unsupported", nil)
	}
}

func (OpenAIChatCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if request.PreviousResponseID != "" || request.ConversationID != "" || request.Store != nil {
		if err := appendLossy(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIChatV1, "conversation_state", "Chat Completions has no stateful response reference"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := chatRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		ParallelToolCalls: request.ParallelToolCalls, CandidateCount: request.CandidateCount,
		Temperature: request.Sampling.Temperature, TopP: request.Sampling.TopP,
		MaxCompletionTokens: request.Sampling.MaxOutputTokens, Seed: request.Sampling.Seed,
		FrequencyPenalty: request.Sampling.FrequencyPenalty, PresencePenalty: request.Sampling.PresencePenalty,
		ReasoningEffort: request.ReasoningEffort, ReasoningBudget: request.ReasoningBudgetTokens,
	}
	if request.Stream {
		wire.StreamOptions = &chatStreamOptionsWire{IncludeUsage: true}
	}
	for _, instruction := range request.Instructions {
		encoded, err := encodeChatMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content})
		if err != nil {
			return nil, diagnostics, err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	for _, message := range request.Messages {
		encoded, err := encodeChatMessage(message)
		if err != nil {
			return nil, nil, err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	for _, tool := range request.Tools {
		wire.Tools = append(wire.Tools, chatToolWire{Type: "function", Function: chatFunctionWire{
			Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict,
		}})
	}
	wire.ToolChoice = encodeChatToolChoice(request.ToolChoice)
	if len(request.Sampling.Stop) == 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop[0])
	} else if len(request.Sampling.Stop) > 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop)
	}
	output, err := encodeChatOutputFormat(request.OutputFormat)
	if err != nil {
		return nil, nil, err
	}
	wire.ResponseFormat = output
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeChatMessage(message llmprotocol.Message) (chatMessageWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return chatMessageWire{}, err
	}
	wire := chatMessageWire{ID: message.ID, Role: role}
	parts := make([]chatContentWire, 0, len(message.Content))
	citationTextBlocks := 0
	for _, content := range message.Content {
		switch content.Kind {
		case llmprotocol.ContentText:
			parts = append(parts, chatContentWire{Type: "text", Text: content.Text})
			if len(content.Citations) > 0 {
				citationTextBlocks++
				wire.Annotations = append(wire.Annotations, encodeChatAnnotations(content.Citations)...)
			}
		case llmprotocol.ContentRefusal:
			wire.Refusal = content.Text
		case llmprotocol.ContentReasoning:
			wire.Reasoning = content.Text
		case llmprotocol.ContentImage:
			imageURL := content.URL
			if content.FileID != "" {
				return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "image_file_id", "Chat Completions cannot encode image file IDs", nil)
			}
			if content.Data != "" {
				if !strings.HasPrefix(content.MediaType, "image/") {
					return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_media_type", "inline Chat images require an image media type", nil)
				}
				imageURL = "data:" + content.MediaType + ";base64," + content.Data
			}
			if imageURL == "" {
				return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_source", "Chat images require a URL or inline data", nil)
			}
			parts = append(parts, chatContentWire{Type: "image_url", ImageURL: &chatImageURLWire{URL: imageURL, Detail: content.Detail}})
		case llmprotocol.ContentAudio:
			parts = append(parts, chatContentWire{Type: "input_audio", InputAudio: &chatInputAudioWire{Data: content.Data, Format: content.MediaType}})
		case llmprotocol.ContentToolCall:
			if content.ToolCall == nil {
				return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call content is invalid", nil)
			}
			wire.ToolCalls = append(wire.ToolCalls, chatToolCallWire{ID: content.ToolCall.ID, Type: "function", Function: chatFunctionWire{Name: content.ToolCall.Name, Arguments: content.ToolCall.Arguments}})
		case llmprotocol.ContentToolResult:
			if content.ToolResult == nil {
				return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result content is invalid", nil)
			}
			wire.ToolCallID = content.ToolResult.CallID
			for _, resultContent := range content.ToolResult.Content {
				if resultContent.Kind != llmprotocol.ContentText {
					return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "tool_result_media", "chat tool results support text only", nil)
				}
				parts = append(parts, chatContentWire{Type: "text", Text: resultContent.Text})
			}
		default:
			return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as chat", nil)
		}
	}
	if citationTextBlocks > 1 {
		return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "citation_text_ambiguous", "Chat Completions citations require one text block", nil)
	}
	if len(parts) == 1 && parts[0].Type == "text" {
		wire.Content, _ = json.Marshal(parts[0].Text)
	} else if len(parts) > 0 {
		wire.Content, _ = json.Marshal(parts)
	}
	return wire, nil
}

func encodeChatToolChoice(choice llmprotocol.ToolChoice) json.RawMessage {
	switch choice.Mode {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
		body, _ := json.Marshal(choice.Mode)
		return body
	case llmprotocol.ToolChoiceNamed:
		body, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]string{"name": choice.Name}})
		return body
	default:
		return nil
	}
}

func encodeChatOutputFormat(format llmprotocol.OutputFormat) (*chatOutputWire, error) {
	switch format.Kind {
	case "", llmprotocol.OutputText:
		return nil, nil
	case llmprotocol.OutputJSONObject:
		return &chatOutputWire{Type: "json_object"}, nil
	case llmprotocol.OutputJSONSchema:
		body, err := json.Marshal(map[string]any{"name": format.Name, "description": format.Description, "strict": format.Strict, "schema": format.Schema})
		return &chatOutputWire{Type: "json_schema", JSONObject: body}, err
	default:
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "output format cannot be encoded as chat", nil)
	}
}

func (OpenAIChatCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire chatResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateChatExecutionMetadata(wire.SystemFingerprint); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.Created > 0 {
		response.CreatedAt = time.Unix(wire.Created, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: wire.Error.Code, Message: wire.Error.Message, Parameter: wire.Error.Param}
		response.StopReason = llmprotocol.StopError
	}
	var diagnostics llmprotocol.Diagnostics
	for choiceIndex, choice := range wire.Choices {
		if err := validateChatChoiceExtensions(choice); err != nil {
			return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
		}
		message, err := decodeChatMessage(choice.Message, choice.Index, policy)
		if err != nil {
			return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
		}
		item := llmprotocol.OutputItem(message)
		if item.ID == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			item.ID = llmprotocol.StableID("chat-response", wire.ID, fmt.Sprint(choice.Index))
		}
		if choiceIndex == 0 {
			response.Output = append(response.Output, item)
			response.Evidence.TokenLogprobs = decodeChatTokenLogprobs(choice.Logprobs)
			if choice.FinishReason != nil {
				response.SourceStopReason = *choice.FinishReason
				response.StopReason = decodeChatStop(*choice.FinishReason)
			}
		} else {
			response.Alternatives = append(response.Alternatives, []llmprotocol.OutputItem{item})
		}
	}
	if wire.Usage != nil {
		response.Usage = decodeChatUsage(*wire.Usage)
	}
	envelope := responseEnvelope(llmprotocol.OpenAIChatV1, body, response.Generation, response.SourceStopReason, policy)
	return response, envelope, diagnostics, nil
}

func decodeChatTokenLogprobs(wire *chatLogprobsWire) []llmprotocol.TokenLogprob {
	if wire == nil {
		return nil
	}
	tokens := make([]llmprotocol.TokenLogprob, 0, len(wire.Content))
	for _, token := range wire.Content {
		decoded := llmprotocol.TokenLogprob{Token: token.Token, Logprob: token.Logprob}
		for _, alternative := range token.TopLogprobs {
			decoded.Alternatives = append(decoded.Alternatives, llmprotocol.TokenLogprobAlternative{
				Token: alternative.Token, Logprob: alternative.Logprob,
			})
		}
		tokens = append(tokens, decoded)
	}
	return tokens
}

func decodeChatUsage(wire chatUsageWire) llmprotocol.Usage {
	usage := llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputUncached:   unknownCount(),
		InputCacheRead:  unknownCount(),
		InputCacheWrite: unknownCount(),
		OutputReasoning: unknownCount(),
		OutputOther:     unknownCount(),
		InputTotal:      authoritative(wire.PromptTokens),
		OutputTotal:     authoritative(wire.CompletionTokens),
		Total:           authoritative(wire.TotalTokens),
	}
	if wire.PromptTokensDetails != nil {
		cached := wire.PromptTokensDetails.CachedTokens
		uncached := wire.PromptTokens - cached
		if cached < 0 || wire.PromptTokens < cached {
			uncached = -1
		}
		usage.InputCacheRead = authoritative(cached)
		usage.InputUncached = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(uncached), Provenance: llmprotocol.UsageDerived,
		}
	}
	if wire.CompletionTokensDetails != nil {
		reasoning := wire.CompletionTokensDetails.ReasoningTokens
		other := wire.CompletionTokens - reasoning
		if reasoning < 0 || wire.CompletionTokens < reasoning {
			other = -1
		}
		usage.OutputReasoning = authoritative(reasoning)
		usage.OutputOther = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(other), Provenance: llmprotocol.UsageDerived,
		}
	}
	return usage
}

func (OpenAIChatCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	if response.Error != nil {
		return OpenAIChatCodec{}.EncodeError(response.Error), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if response.Usage.InputCacheWrite.Value != nil {
		appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIChatV1, "usage.input_cache_write", "Chat Completions has no cache-write usage field")
	}
	wire := chatResponseWire{ID: response.ID, Object: "chat.completion", Model: response.Model}
	if !response.CreatedAt.IsZero() {
		wire.Created = response.CreatedAt.Unix()
	}
	primary, encodeResponseErr := encodeChatOutput(response.Output, 0, response.StopReason)
	if encodeResponseErr != nil {
		return nil, diagnostics, encodeResponseErr
	}
	wire.Choices = append(wire.Choices, primary)
	for index, alternative := range response.Alternatives {
		choice, err := encodeChatOutput(alternative, index+1, response.StopReason)
		if err != nil {
			return nil, diagnostics, err
		}
		wire.Choices = append(wire.Choices, choice)
	}
	wire.Usage = encodeChatUsage(response.Usage)
	body, encodeResponseErr := marshalWire(wire)
	return body, diagnostics, encodeResponseErr
}

func encodeChatOutput(items []llmprotocol.OutputItem, index int, stop llmprotocol.StopReason) (chatChoiceWire, error) {
	combined := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, item := range items {
		if item.Role != "" {
			combined.Role = item.Role
		}
		combined.Content = append(combined.Content, item.Content...)
	}
	message, err := encodeChatMessage(combined)
	if err != nil {
		return chatChoiceWire{}, err
	}
	reason := encodeChatStop(stop)
	return chatChoiceWire{Index: index, Message: message, FinishReason: &reason}, nil
}

func encodeChatUsage(usage llmprotocol.Usage) *chatUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	prompt := tokenValue(usage.InputTotal)
	completion := tokenValue(usage.OutputTotal)
	total := tokenValue(usage.Total)
	if total == 0 {
		total = prompt + completion
	}
	wire := &chatUsageWire{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: total}
	if usage.InputCacheRead.Value != nil {
		wire.PromptTokensDetails = &chatPromptTokensDetailsWire{CachedTokens: tokenValue(usage.InputCacheRead)}
	}
	if usage.OutputReasoning.Value != nil {
		wire.CompletionTokensDetails = &chatCompletionTokensDetailsWire{ReasoningTokens: tokenValue(usage.OutputReasoning)}
	}
	return wire
}

func decodeChatStop(reason string) llmprotocol.StopReason {
	switch reason {
	case "stop":
		return llmprotocol.StopEndTurn
	case "length":
		return llmprotocol.StopMaxTokens
	case "tool_calls", "function_call":
		return llmprotocol.StopToolCall
	case "content_filter":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeChatStop(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "length"
	case llmprotocol.StopToolCall:
		return "tool_calls"
	case llmprotocol.StopContentFilter:
		return "content_filter"
	default:
		return "stop"
	}
}

func (OpenAIChatCodec) EncodeError(protocolError *llmprotocol.ProtocolError) []byte {
	wire := chatResponseWire{Error: &chatErrorWire{Message: protocolError.Message, Type: string(protocolError.Category), Param: protocolError.Parameter, Code: protocolError.Code}}
	body, _ := json.Marshal(wire)
	return body
}
