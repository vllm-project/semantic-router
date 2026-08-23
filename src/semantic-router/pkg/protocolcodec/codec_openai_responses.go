package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type OpenAIResponsesCodec struct{}

func (OpenAIResponsesCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIResponsesV1 }
func (OpenAIResponsesCodec) Stateless() bool                { return true }
func (OpenAIResponsesCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStructuredJSON, llmprotocol.CapabilityStrictJSONSchema, llmprotocol.CapabilityStrictToolSchema,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
	)
}

type responsesRequestWire struct {
	Model              string                  `json:"model"`
	Input              json.RawMessage         `json:"input"`
	Instructions       json.RawMessage         `json:"instructions,omitempty"`
	Tools              []responsesToolWire     `json:"tools,omitempty"`
	ToolChoice         json.RawMessage         `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool                   `json:"parallel_tool_calls,omitempty"`
	Temperature        *float64                `json:"temperature,omitempty"`
	TopP               *float64                `json:"top_p,omitempty"`
	MaxOutputTokens    *int64                  `json:"max_output_tokens,omitempty"`
	Metadata           map[string]string       `json:"metadata,omitempty"`
	Text               *responsesTextWire      `json:"text,omitempty"`
	Stream             bool                    `json:"stream,omitempty"`
	Store              *bool                   `json:"store,omitempty"`
	PreviousResponseID string                  `json:"previous_response_id,omitempty"`
	ConversationID     string                  `json:"conversation_id,omitempty"`
	AutoStore          *bool                   `json:"auto_store,omitempty"`
	Reasoning          *responsesReasoningWire `json:"reasoning,omitempty"`
	Truncation         json.RawMessage         `json:"truncation,omitempty"`
	User               string                  `json:"user,omitempty"`
}

type responsesReasoningWire struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type responsesTextWire struct {
	Format responsesFormatWire `json:"format,omitempty"`
}

type responsesFormatWire struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type responsesToolWire struct {
	Type        string            `json:"type"`
	Name        string            `json:"name,omitempty"`
	Description string            `json:"description,omitempty"`
	Parameters  json.RawMessage   `json:"parameters,omitempty"`
	Strict      *bool             `json:"strict,omitempty"`
	Function    *chatFunctionWire `json:"function,omitempty"`
}

type responsesItemWire struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Role      string          `json:"role,omitempty"`
	Status    string          `json:"status,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	Name      string          `json:"name,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    json.RawMessage `json:"output,omitempty"`
	Summary   json.RawMessage `json:"summary,omitempty"`
}

type responsesContentWire struct {
	Type        string                    `json:"type"`
	Text        string                    `json:"text,omitempty"`
	Refusal     string                    `json:"refusal,omitempty"`
	Annotations []responsesAnnotationWire `json:"annotations,omitempty"`
	ImageURL    string                    `json:"image_url,omitempty"`
	FileID      string                    `json:"file_id,omitempty"`
	FileData    string                    `json:"file_data,omitempty"`
	Filename    string                    `json:"filename,omitempty"`
	Detail      string                    `json:"detail,omitempty"`
}

type responsesAnnotationWire struct {
	Type       string `json:"type"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex int64  `json:"start_index"`
	EndIndex   int64  `json:"end_index"`
}

func (OpenAIResponsesCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire responsesRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		PreviousResponseID: wire.PreviousResponseID, ConversationID: wire.ConversationID,
		Store: wire.Store, AutoStore: wire.AutoStore, ParallelToolCalls: wire.ParallelToolCalls,
		Sampling: llmprotocol.Sampling{Temperature: wire.Temperature, TopP: wire.TopP, MaxOutputTokens: wire.MaxOutputTokens},
		Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIResponsesV1},
	}
	if len(wire.Instructions) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Instructions), []byte("null")) {
		instructions, err := decodeResponsesContent(wire.Instructions, policy, true)
		if err != nil {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
		}
		request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleDeveloper, Content: instructions}}
	}
	if err := decodeResponsesInput(wire.Input, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	for _, tool := range wire.Tools {
		if tool.Type != "function" {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools enter the model protocol", nil)
		}
		name, description, schema, strict := tool.Name, tool.Description, tool.Parameters, tool.Strict
		if tool.Function != nil {
			name, description, schema, strict = tool.Function.Name, tool.Function.Description, tool.Function.Parameters, tool.Function.Strict
		}
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{Name: name, Description: description, InputSchema: schema, Strict: strict})
	}
	choice, err := decodeResponsesToolChoice(wire.ToolChoice)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request.ToolChoice = choice
	if wire.Text != nil {
		request.OutputFormat = llmprotocol.OutputFormat{
			Kind: llmprotocol.OutputFormatKind(wire.Text.Format.Type), Name: wire.Text.Format.Name,
			Description: wire.Text.Format.Description, Strict: wire.Text.Format.Strict, Schema: wire.Text.Format.Schema,
		}
		switch request.OutputFormat.Kind {
		case "json_schema":
			request.OutputFormat.Kind = llmprotocol.OutputJSONSchema
		case "json_object":
			request.OutputFormat.Kind = llmprotocol.OutputJSONObject
		case "", "text":
			request.OutputFormat.Kind = llmprotocol.OutputText
		default:
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "Responses output format is unsupported", nil)
		}
	}
	return request, requestEnvelope(llmprotocol.OpenAIResponsesV1, body, request.Generation, policy), nil, nil
}

func decodeResponsesInput(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "input_required", "input is required", nil)
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}})
		return nil
	}
	var items []responsesItemWire
	if err := decodeWire(raw, &items, policy); err != nil {
		return err
	}
	for index, item := range items {
		switch item.Type {
		case "", "message":
			role, err := canonicalRole(item.Role)
			if err != nil {
				return err
			}
			content, err := decodeResponsesContent(item.Content, policy, false)
			if err != nil {
				return err
			}
			if role == llmprotocol.RoleSystem || role == llmprotocol.RoleDeveloper {
				request.Instructions = append(request.Instructions, llmprotocol.InstructionBlock{Role: role, Content: content})
			} else {
				request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: role, Content: content})
			}
		case "function_call":
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
				id = llmprotocol.StableID("responses", fmt.Sprint(index), item.Name, item.Arguments)
			}
			request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: item.Name, Arguments: item.Arguments}}}})
		case "function_call_output":
			content, err := decodeResponsesContent(item.Output, policy, false)
			if err != nil {
				return err
			}
			request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: item.CallID, Content: content}}}})
		case "reasoning":
			content, err := decodeResponsesReasoning(item.Summary, policy)
			if err != nil {
				return err
			}
			request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: llmprotocol.RoleAssistant, Content: content})
		case "item_reference":
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unresolved_item_reference", "item references must be resolved before model dispatch", nil)
		default:
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_input_item", "Responses input item is unsupported", nil)
		}
	}
	return nil
}

func decodeResponsesContent(raw json.RawMessage, policy llmprotocol.Policy, instruction bool) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
	}
	var parts []responsesContentWire
	if err := decodeWire(raw, &parts, policy); err != nil {
		return nil, err
	}
	result := make([]llmprotocol.Content, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "text":
			citations, err := decodeResponsesAnnotations(part.Annotations)
			if err != nil {
				return nil, err
			}
			result = append(result, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text, Citations: citations})
		case "refusal":
			result = append(result, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.Refusal})
		case "input_image":
			if instruction {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "instruction_image", "instructions cannot contain images", nil)
			}
			if mediaType, data, inline := decodeDataURL(part.ImageURL); inline {
				result = append(result, llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: mediaType, Data: data, Detail: part.Detail})
			} else {
				result = append(result, llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: part.ImageURL, FileID: part.FileID, Detail: part.Detail})
			}
		case "input_file":
			result = append(result, llmprotocol.Content{Kind: llmprotocol.ContentFile, FileID: part.FileID, Data: part.FileData, Filename: part.Filename})
		default:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Responses content type is unsupported", nil)
		}
	}
	return result, nil
}

func decodeResponsesAnnotations(wire []responsesAnnotationWire) ([]llmprotocol.Citation, error) {
	citations := make([]llmprotocol.Citation, 0, len(wire))
	for _, annotation := range wire {
		if annotation.Type != "url_citation" {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_annotation", "Responses annotation is unsupported", nil)
		}
		citations = append(citations, llmprotocol.Citation{
			URL: annotation.URL, Title: annotation.Title,
			StartIndex: annotation.StartIndex, EndIndex: annotation.EndIndex,
		})
	}
	return citations, nil
}

func encodeResponsesAnnotations(citations []llmprotocol.Citation) []responsesAnnotationWire {
	annotations := make([]responsesAnnotationWire, 0, len(citations))
	for _, citation := range citations {
		annotations = append(annotations, responsesAnnotationWire{
			Type: "url_citation", URL: citation.URL, Title: citation.Title,
			StartIndex: citation.StartIndex, EndIndex: citation.EndIndex,
		})
	}
	return annotations
}

func decodeResponsesToolChoice(raw json.RawMessage) (llmprotocol.ToolChoice, error) {
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
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &named) != nil || named.Type != "function" || named.Name == "" {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Responses tool choice is invalid", nil)
	}
	return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: named.Name}, nil
}

func (OpenAIResponsesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	wire := responsesRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		Store: request.Store, AutoStore: request.AutoStore, PreviousResponseID: request.PreviousResponseID, ConversationID: request.ConversationID,
		ParallelToolCalls: request.ParallelToolCalls, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, MaxOutputTokens: request.Sampling.MaxOutputTokens,
	}
	items := make([]responsesItemWire, 0, len(request.Messages))
	for _, instruction := range request.Instructions {
		encoded, err := encodeResponsesMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content}, "input")
		if err != nil {
			return nil, nil, err
		}
		items = append(items, encoded...)
	}
	for _, message := range request.Messages {
		encoded, err := encodeResponsesMessage(message, "input")
		if err != nil {
			return nil, nil, err
		}
		items = append(items, encoded...)
	}
	wire.Input, _ = json.Marshal(items)
	for _, tool := range request.Tools {
		wire.Tools = append(wire.Tools, responsesToolWire{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict})
	}
	wire.ToolChoice = encodeResponsesToolChoice(request.ToolChoice)
	if request.OutputFormat.Kind != "" && request.OutputFormat.Kind != llmprotocol.OutputText {
		wire.Text = &responsesTextWire{Format: responsesFormatWire{
			Type: string(request.OutputFormat.Kind), Name: request.OutputFormat.Name,
			Description: request.OutputFormat.Description, Strict: request.OutputFormat.Strict, Schema: request.OutputFormat.Schema,
		}}
	}
	body, err := marshalWire(wire)
	return body, nil, err
}

func encodeResponsesMessage(message llmprotocol.Message, textDirection string) ([]responsesItemWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return nil, err
	}
	ordinary := make([]llmprotocol.Content, 0, len(message.Content))
	items := make([]responsesItemWire, 0)
	flush := func() error {
		if len(ordinary) == 0 {
			return nil
		}
		content, err := encodeResponsesContent(ordinary, textDirection)
		if err != nil {
			return err
		}
		items = append(items, responsesItemWire{Type: "message", ID: responsesItemID(message.ID, len(items), "message"), Role: role, Content: content})
		ordinary = nil
		return nil
	}
	for _, content := range message.Content {
		switch content.Kind {
		case llmprotocol.ContentToolCall:
			if err := flush(); err != nil {
				return nil, err
			}
			if content.ToolCall == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call is invalid", nil)
			}
			items = append(items, responsesItemWire{Type: "function_call", ID: responsesItemID(message.ID, len(items), "function_call"), CallID: content.ToolCall.ID, Name: content.ToolCall.Name, Arguments: content.ToolCall.Arguments})
		case llmprotocol.ContentToolResult:
			if err := flush(); err != nil {
				return nil, err
			}
			if content.ToolResult == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result is invalid", nil)
			}
			output, err := encodeResponsesContent(content.ToolResult.Content, "output")
			if err != nil {
				return nil, err
			}
			items = append(items, responsesItemWire{Type: "function_call_output", ID: responsesItemID(message.ID, len(items), "function_call_output"), CallID: content.ToolResult.CallID, Output: output})
		case llmprotocol.ContentReasoning:
			if err := flush(); err != nil {
				return nil, err
			}
			summary, _ := json.Marshal([]map[string]string{{"type": "summary_text", "text": content.Text}})
			items = append(items, responsesItemWire{Type: "reasoning", ID: responsesItemID(message.ID, len(items), "reasoning"), Summary: summary})
		default:
			ordinary = append(ordinary, content)
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return items, nil
}

func responsesItemID(messageID string, index int, kind string) string {
	if index == 0 && messageID != "" {
		return messageID
	}
	return llmprotocol.StableID("responses-item", messageID, fmt.Sprint(index), kind)
}

func decodeResponsesReasoning(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Content, error) {
	var summaries []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := decodeWire(raw, &summaries, policy); err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(summaries))
	for _, summary := range summaries {
		if summary.Type != "summary_text" {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_reasoning_summary", "Responses reasoning summary is unsupported", nil)
		}
		contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: summary.Text})
	}
	return contents, nil
}

func encodeResponsesContent(contents []llmprotocol.Content, direction string) (json.RawMessage, error) {
	parts := make([]responsesContentWire, 0, len(contents))
	for _, content := range contents {
		switch content.Kind {
		case llmprotocol.ContentText:
			parts = append(parts, responsesContentWire{Type: direction + "_text", Text: content.Text, Annotations: encodeResponsesAnnotations(content.Citations)})
		case llmprotocol.ContentRefusal:
			parts = append(parts, responsesContentWire{Type: "refusal", Refusal: content.Text})
		case llmprotocol.ContentReasoning:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "reasoning_content_position", "Responses reasoning must be encoded as an ordered reasoning item", nil)
		case llmprotocol.ContentImage:
			parts = append(parts, responsesContentWire{Type: "input_image", ImageURL: content.URL, FileID: content.FileID, Detail: content.Detail})
		case llmprotocol.ContentFile:
			parts = append(parts, responsesContentWire{Type: "input_file", FileID: content.FileID, FileData: content.Data, Filename: content.Filename})
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
	default:
		return nil
	}
}

type responsesResponseWire struct {
	ID                string              `json:"id"`
	Object            string              `json:"object,omitempty"`
	CreatedAt         int64               `json:"created_at,omitempty"`
	Model             string              `json:"model"`
	Status            string              `json:"status,omitempty"`
	Output            []responsesItemWire `json:"output,omitempty"`
	OutputText        string              `json:"output_text,omitempty"`
	Usage             *responsesUsageWire `json:"usage,omitempty"`
	Error             *responsesErrorWire `json:"error,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	ConversationID     string            `json:"conversation_id,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
}

type responsesUsageWire struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details,omitempty"`
}

type responsesErrorWire struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (OpenAIResponsesCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire responsesResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.CreatedAt > 0 {
		response.CreatedAt = time.Unix(wire.CreatedAt, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: wire.Error.Code, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	for index, item := range wire.Output {
		output, err := decodeResponsesOutputItem(item, index, policy)
		if err != nil {
			return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
		}
		response.Output = append(response.Output, output)
	}
	if wire.Usage != nil {
		response.Usage = decodeResponsesUsage(*wire.Usage)
	}
	if response.Error == nil {
		response.StopReason = llmprotocol.StopEndTurn
	}
	response.SourceStopReason = wire.Status
	if wire.Status == "incomplete" && wire.IncompleteDetails != nil && wire.IncompleteDetails.Reason == "max_output_tokens" {
		response.StopReason = llmprotocol.StopMaxTokens
	}
	if wire.Status == "failed" {
		response.StopReason = llmprotocol.StopError
	}
	return response, responseEnvelope(llmprotocol.OpenAIResponsesV1, body, response.Generation, wire.Status, policy), nil, nil
}

func decodeResponsesOutputItem(item responsesItemWire, index int, policy llmprotocol.Policy) (llmprotocol.OutputItem, error) {
	id := item.ID
	if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		id = llmprotocol.StableID("response-output", fmt.Sprint(index), item.Type, item.CallID)
	}
	output := llmprotocol.OutputItem{ID: id, Role: llmprotocol.RoleAssistant}
	switch item.Type {
	case "message":
		role, err := canonicalRole(item.Role)
		if err != nil {
			return llmprotocol.OutputItem{}, err
		}
		content, err := decodeResponsesContent(item.Content, policy, false)
		if err != nil {
			return llmprotocol.OutputItem{}, err
		}
		output.Role, output.Content = role, content
	case "function_call":
		output.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}}}
	case "reasoning":
		content, err := decodeResponsesReasoning(item.Summary, policy)
		if err != nil {
			return llmprotocol.OutputItem{}, err
		}
		output.Content = content
	default:
		return llmprotocol.OutputItem{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses output item is unsupported", nil)
	}
	return output, nil
}

func decodeResponsesUsage(wire responsesUsageWire) llmprotocol.Usage {
	cached := int64(0)
	if wire.InputTokensDetails != nil {
		cached = wire.InputTokensDetails.CachedTokens
	}
	reasoning := int64(0)
	if wire.OutputTokensDetails != nil {
		reasoning = wire.OutputTokensDetails.ReasoningTokens
	}
	uncached, other := wire.InputTokens-cached, wire.OutputTokens-reasoning
	if uncached < 0 {
		uncached = 0
	}
	if other < 0 {
		other = 0
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(uncached), InputCacheRead: authoritative(cached), InputCacheWrite: unknownCount(),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(wire.InputTokens), OutputTotal: authoritative(wire.OutputTokens), Total: authoritative(wire.TotalTokens),
	}
}

func (OpenAIResponsesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	if response.Error != nil {
		return OpenAIResponsesCodec{}.EncodeError(response.Error), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if response.Usage.InputCacheWrite.Value != nil {
		appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIResponsesV1, "usage.input_cache_write", "Responses has no cache-write usage field")
	}
	if len(response.Alternatives) > 0 {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIResponsesV1, "response.alternatives", "Responses has one primary output sequence"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := responsesResponseWire{ID: response.ID, Object: "response", Model: response.Model, Status: "completed"}
	if !response.CreatedAt.IsZero() {
		wire.CreatedAt = response.CreatedAt.Unix()
	}
	for _, item := range response.Output {
		encoded, err := encodeResponsesOutputItem(item)
		if err != nil {
			return nil, diagnostics, err
		}
		wire.Output = append(wire.Output, encoded...)
	}
	wire.Usage = encodeResponsesUsage(response.Usage)
	if response.StopReason == llmprotocol.StopMaxTokens {
		wire.Status = "incomplete"
		wire.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: "max_output_tokens"}
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeResponsesOutputItem(item llmprotocol.OutputItem) ([]responsesItemWire, error) {
	message := llmprotocol.Message(item)
	return encodeResponsesMessage(message, "output")
}

func encodeResponsesUsage(usage llmprotocol.Usage) *responsesUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	wire := &responsesUsageWire{InputTokens: tokenValue(usage.InputTotal), OutputTokens: tokenValue(usage.OutputTotal), TotalTokens: tokenValue(usage.Total)}
	if wire.TotalTokens == 0 {
		wire.TotalTokens = wire.InputTokens + wire.OutputTokens
	}
	if usage.InputCacheRead.Value != nil {
		wire.InputTokensDetails = &struct {
			CachedTokens int64 `json:"cached_tokens"`
		}{CachedTokens: tokenValue(usage.InputCacheRead)}
	}
	if usage.OutputReasoning.Value != nil {
		wire.OutputTokensDetails = &struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		}{ReasoningTokens: tokenValue(usage.OutputReasoning)}
	}
	return wire
}

func (OpenAIResponsesCodec) EncodeError(protocolError *llmprotocol.ProtocolError) []byte {
	wire := responsesResponseWire{Object: "error", Status: "failed", Error: &responsesErrorWire{Code: protocolError.Code, Message: protocolError.Message}}
	body, _ := json.Marshal(wire)
	return body
}
