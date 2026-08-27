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
	Model                string                  `json:"model"`
	Input                json.RawMessage         `json:"input"`
	Instructions         json.RawMessage         `json:"instructions,omitempty"`
	Tools                []responsesToolWire     `json:"tools,omitempty"`
	ToolChoice           json.RawMessage         `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                   `json:"parallel_tool_calls,omitempty"`
	Temperature          *float64                `json:"temperature,omitempty"`
	TopP                 *float64                `json:"top_p,omitempty"`
	MaxOutputTokens      *int64                  `json:"max_output_tokens,omitempty"`
	Metadata             map[string]string       `json:"metadata,omitempty"`
	Text                 *responsesTextWire      `json:"text,omitempty"`
	Stream               bool                    `json:"stream,omitempty"`
	Store                *bool                   `json:"store,omitempty"`
	PreviousResponseID   string                  `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage         `json:"conversation,omitempty"`
	AutoStore            *bool                   `json:"auto_store,omitempty"`
	Reasoning            *responsesReasoningWire `json:"reasoning,omitempty"`
	Truncation           string                  `json:"truncation,omitempty"`
	User                 string                  `json:"user,omitempty"`
	Background           json.RawMessage         `json:"background,omitempty"`
	ContextManagement    json.RawMessage         `json:"context_management,omitempty"`
	Include              json.RawMessage         `json:"include,omitempty"`
	MaxToolCalls         json.RawMessage         `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage         `json:"moderation,omitempty"`
	Prompt               json.RawMessage         `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage         `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention json.RawMessage         `json:"prompt_cache_retention,omitempty"`
	PromptCacheOptions   json.RawMessage         `json:"prompt_cache_options,omitempty"`
	SafetyIdentifier     json.RawMessage         `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage         `json:"service_tier,omitempty"`
	StreamOptions        json.RawMessage         `json:"stream_options,omitempty"`
	TopLogprobs          json.RawMessage         `json:"top_logprobs,omitempty"`
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
	Type           string          `json:"type"`
	Name           string          `json:"name,omitempty"`
	Description    string          `json:"description,omitempty"`
	Parameters     json.RawMessage `json:"parameters,omitempty"`
	Strict         *bool           `json:"strict,omitempty"`
	AllowedCallers json.RawMessage `json:"allowed_callers,omitempty"`
	DeferLoading   json.RawMessage `json:"defer_loading,omitempty"`
	OutputSchema   json.RawMessage `json:"output_schema,omitempty"`
}

type responsesItemWire struct {
	Type             string          `json:"type"`
	ID               string          `json:"id,omitempty"`
	Role             string          `json:"role,omitempty"`
	Status           string          `json:"status,omitempty"`
	Content          json.RawMessage `json:"content,omitempty"`
	Name             string          `json:"name,omitempty"`
	CallID           string          `json:"call_id,omitempty"`
	Arguments        string          `json:"arguments,omitempty"`
	Output           json.RawMessage `json:"output,omitempty"`
	Summary          json.RawMessage `json:"summary,omitempty"`
	Phase            json.RawMessage `json:"phase,omitempty"`
	EncryptedContent json.RawMessage `json:"encrypted_content,omitempty"`
}

type responsesContentWire struct {
	Type                  string                    `json:"type"`
	Text                  string                    `json:"text,omitempty"`
	Refusal               string                    `json:"refusal,omitempty"`
	Annotations           []responsesAnnotationWire `json:"annotations,omitempty"`
	ImageURL              string                    `json:"image_url,omitempty"`
	FileURL               string                    `json:"file_url,omitempty"`
	FileID                string                    `json:"file_id,omitempty"`
	FileData              string                    `json:"file_data,omitempty"`
	Filename              string                    `json:"filename,omitempty"`
	Detail                string                    `json:"detail,omitempty"`
	Logprobs              json.RawMessage           `json:"logprobs,omitempty"`
	PromptCacheBreakpoint json.RawMessage           `json:"prompt_cache_breakpoint,omitempty"`
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
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"background": wire.Background, "context_management": wire.ContextManagement,
		"include": wire.Include, "max_tool_calls": wire.MaxToolCalls, "moderation": wire.Moderation,
		"prompt": wire.Prompt, "prompt_cache_key": wire.PromptCacheKey,
		"prompt_cache_retention": wire.PromptCacheRetention,
		"prompt_cache_options":   wire.PromptCacheOptions, "safety_identifier": wire.SafetyIdentifier,
		"service_tier": wire.ServiceTier, "stream_options": wire.StreamOptions,
		"top_logprobs": wire.TopLogprobs,
	}); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	conversationID, err := decodeResponsesConversation(wire.Conversation)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request := decodeResponsesBaseRequest(wire, conversationID)
	if err := decodeResponsesReasoningRequest(wire.Reasoning, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesInstructions(wire.Instructions, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesInput(wire.Input, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesTools(wire.Tools, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesRequestOptions(wire, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.OpenAIResponsesV1, body, request.Generation, policy), nil, nil
}

func decodeResponsesBaseRequest(wire responsesRequestWire, conversationID string) llmprotocol.Request {
	return llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		EndUserID: wire.User, PreviousResponseID: wire.PreviousResponseID, ConversationID: conversationID,
		Truncation: wire.Truncation,
		Store:      wire.Store, AutoStore: wire.AutoStore, ParallelToolCalls: wire.ParallelToolCalls,
		Sampling: llmprotocol.Sampling{Temperature: wire.Temperature, TopP: wire.TopP, MaxOutputTokens: wire.MaxOutputTokens},
		Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIResponsesV1},
	}
}

func decodeResponsesReasoningRequest(reasoning *responsesReasoningWire, request *llmprotocol.Request) error {
	if reasoning == nil {
		return nil
	}
	request.ReasoningEffort = reasoning.Effort
	if reasoning.Summary != "" {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_reasoning_summary",
			"reasoning.summary is not supported by the protocol-neutral request contract", nil,
		)
	}
	return nil
}

func decodeResponsesInstructions(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	instructions, err := decodeResponsesContent(raw, policy, true, false)
	if err != nil {
		return err
	}
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleDeveloper, Content: instructions}}
	return nil
}

func decodeResponsesTools(tools []responsesToolWire, request *llmprotocol.Request) error {
	for _, tool := range tools {
		if tool.Type != "function" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools enter the model protocol", nil)
		}
		if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
			"tools.allowed_callers": tool.AllowedCallers,
			"tools.defer_loading":   tool.DeferLoading,
			"tools.output_schema":   tool.OutputSchema,
		}); err != nil {
			return err
		}
		name, description, schema, strict := tool.Name, tool.Description, tool.Parameters, tool.Strict
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{Name: name, Description: description, InputSchema: schema, Strict: strict})
	}
	return nil
}

func decodeResponsesRequestOptions(wire responsesRequestWire, request *llmprotocol.Request) error {
	choice, err := decodeResponsesToolChoice(wire.ToolChoice)
	if err != nil {
		return err
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
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "Responses output format is unsupported", nil)
		}
	}
	return nil
}

func decodeResponsesConversation(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	var id string
	if json.Unmarshal(raw, &id) == nil {
		return id, nil
	}
	var object struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(raw, &object) != nil || object.ID == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_conversation", "conversation must be an ID or object with an ID", nil)
	}
	return object.ID, nil
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
		if err := decodeResponsesInputItem(item, index, request, policy); err != nil {
			return err
		}
	}
	return nil
}

func decodeResponsesInputItem(
	item responsesItemWire,
	index int,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"input.encrypted_content": item.EncryptedContent,
		"input.phase":             item.Phase,
	}); err != nil {
		return err
	}
	switch item.Type {
	case "", "message":
		return decodeResponsesMessageItem(item, request, policy)
	case "function_call":
		request.Messages = append(request.Messages, decodeResponsesFunctionCall(item, index, policy))
		return nil
	case "function_call_output":
		return decodeResponsesFunctionResult(item, request, policy)
	case "reasoning":
		return decodeResponsesReasoningItem(item, request, policy)
	case "item_reference":
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unresolved_item_reference", "item references must be resolved before model dispatch", nil)
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_input_item", "Responses input item is unsupported", nil)
	}
}

func decodeResponsesMessageItem(item responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	role, err := canonicalRole(item.Role)
	if err != nil {
		return err
	}
	content, err := decodeResponsesContent(item.Content, policy, false, false)
	if err != nil {
		return err
	}
	if role == llmprotocol.RoleSystem || role == llmprotocol.RoleDeveloper {
		request.Instructions = append(request.Instructions, llmprotocol.InstructionBlock{Role: role, Content: content})
	} else {
		request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: role, Content: content})
	}
	return nil
}

func decodeResponsesFunctionCall(item responsesItemWire, index int, policy llmprotocol.Policy) llmprotocol.Message {
	id := item.CallID
	if id == "" {
		id = item.ID
	}
	if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		id = llmprotocol.StableID("responses", fmt.Sprint(index), item.Name, item.Arguments)
	}
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind:     llmprotocol.ContentToolCall,
		ToolCall: &llmprotocol.ToolCall{ID: id, Name: item.Name, Arguments: item.Arguments},
	}}}
}

func decodeResponsesFunctionResult(item responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	content, err := decodeResponsesContent(item.Output, policy, false, false)
	if err != nil {
		return err
	}
	request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind:       llmprotocol.ContentToolResult,
		ToolResult: &llmprotocol.ToolResult{CallID: item.CallID, Content: content},
	}}})
	return nil
}

func decodeResponsesReasoningItem(item responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	content, err := decodeResponsesReasoning(item.Summary, policy)
	if err != nil {
		return err
	}
	request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: llmprotocol.RoleAssistant, Content: content})
	return nil
}

func decodeResponsesContent(raw json.RawMessage, policy llmprotocol.Policy, instruction, providerOutput bool) ([]llmprotocol.Content, error) {
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
		content, err := decodeResponsesContentPart(part, instruction, providerOutput)
		if err != nil {
			return nil, err
		}
		result = append(result, content)
	}
	return result, nil
}

func decodeResponsesContentPart(
	part responsesContentWire,
	instruction bool,
	providerOutput bool,
) (llmprotocol.Content, error) {
	unsupported := map[string]json.RawMessage{
		"content.prompt_cache_breakpoint": part.PromptCacheBreakpoint,
	}
	if !providerOutput {
		unsupported["content.logprobs"] = part.Logprobs
	}
	if err := rejectUnsupportedRequestFields(unsupported); err != nil {
		return llmprotocol.Content{}, err
	}
	switch part.Type {
	case "input_text", "output_text", "text":
		citations, err := decodeResponsesAnnotations(part.Annotations)
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text, Citations: citations}, err
	case "refusal":
		return llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.Refusal}, nil
	case "input_image":
		return decodeResponsesImage(part, instruction)
	case "input_file":
		return decodeResponsesFile(part), nil
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Responses content type is unsupported", nil)
	}
}

func decodeResponsesImage(part responsesContentWire, instruction bool) (llmprotocol.Content, error) {
	if instruction {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "instruction_image", "instructions cannot contain images", nil)
	}
	if mediaType, data, inline := decodeDataURL(part.ImageURL); inline {
		return llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: mediaType, Data: data, Detail: part.Detail}, nil
	}
	return llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: part.ImageURL, FileID: part.FileID, Detail: part.Detail}, nil
}

func decodeResponsesFile(part responsesContentWire) llmprotocol.Content {
	content := llmprotocol.Content{Kind: llmprotocol.ContentFile, URL: part.FileURL, FileID: part.FileID, Filename: part.Filename}
	if part.FileData == "" {
		return content
	}
	if mediaType, data, inline := decodeDataURL(part.FileData); inline {
		content.MediaType, content.Data = mediaType, data
	} else {
		content.MediaType, content.Data = "application/octet-stream", part.FileData
	}
	return content
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
		Store: request.Store, AutoStore: request.AutoStore, PreviousResponseID: request.PreviousResponseID,
		Truncation: request.Truncation, User: request.EndUserID,
		ParallelToolCalls: request.ParallelToolCalls, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, MaxOutputTokens: request.Sampling.MaxOutputTokens,
	}
	if request.ConversationID != "" {
		wire.Conversation, _ = json.Marshal(request.ConversationID)
	}
	if request.ReasoningEffort != "" {
		wire.Reasoning = &responsesReasoningWire{Effort: request.ReasoningEffort}
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
	state := responsesMessageEncodingState{messageID: message.ID, role: role, textDirection: textDirection}
	for _, content := range message.Content {
		if err := state.appendContent(content); err != nil {
			return nil, err
		}
	}
	if err := state.flushOrdinary(); err != nil {
		return nil, err
	}
	return state.items, nil
}

type responsesMessageEncodingState struct {
	messageID     string
	role          string
	textDirection string
	ordinary      []llmprotocol.Content
	items         []responsesItemWire
}

func (state *responsesMessageEncodingState) appendContent(content llmprotocol.Content) error {
	switch content.Kind {
	case llmprotocol.ContentToolCall:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		return state.appendToolCall(content.ToolCall)
	case llmprotocol.ContentToolResult:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		return state.appendToolResult(content.ToolResult)
	case llmprotocol.ContentReasoning:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		state.appendReasoning(content.Text)
	default:
		state.ordinary = append(state.ordinary, content)
	}
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
	state.items = append(state.items, responsesItemWire{
		Type: "message", ID: responsesItemID(state.messageID, len(state.items), "message"),
		Role: state.role, Content: content,
	})
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
	output, err := encodeResponsesContent(result.Content, "output")
	if err != nil {
		return err
	}
	state.items = append(state.items, responsesItemWire{
		Type: "function_call_output", ID: responsesItemID(state.messageID, len(state.items), "function_call_output"),
		CallID: result.CallID, Output: output,
	})
	return nil
}

func (state *responsesMessageEncodingState) appendReasoning(text string) {
	summary, _ := json.Marshal([]map[string]string{{"type": "summary_text", "text": text}})
	state.items = append(state.items, responsesItemWire{
		Type: "reasoning", ID: responsesItemID(state.messageID, len(state.items), "reasoning"), Summary: summary,
	})
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
			parts = append(parts, responsesContentWire{Type: "input_file", FileURL: content.URL, FileID: content.FileID, FileData: content.Data, Filename: content.Filename})
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
	Usage             *responsesUsageWire `json:"usage,omitempty"`
	Error             *responsesErrorWire `json:"error,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
	PreviousResponseID   string            `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage   `json:"conversation,omitempty"`
	ConversationID       string            `json:"conversation_id,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
	Background           json.RawMessage   `json:"background,omitempty"`
	CompletedAt          json.RawMessage   `json:"completed_at,omitempty"`
	Instructions         json.RawMessage   `json:"instructions,omitempty"`
	MaxOutputTokens      json.RawMessage   `json:"max_output_tokens,omitempty"`
	MaxToolCalls         json.RawMessage   `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage   `json:"moderation,omitempty"`
	OutputText           json.RawMessage   `json:"output_text,omitempty"`
	ParallelToolCalls    json.RawMessage   `json:"parallel_tool_calls,omitempty"`
	Prompt               json.RawMessage   `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage   `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   json.RawMessage   `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention json.RawMessage   `json:"prompt_cache_retention,omitempty"`
	Reasoning            json.RawMessage   `json:"reasoning,omitempty"`
	SafetyIdentifier     json.RawMessage   `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage   `json:"service_tier,omitempty"`
	Store                json.RawMessage   `json:"store,omitempty"`
	Temperature          json.RawMessage   `json:"temperature,omitempty"`
	Text                 json.RawMessage   `json:"text,omitempty"`
	ToolChoice           json.RawMessage   `json:"tool_choice,omitempty"`
	Tools                json.RawMessage   `json:"tools,omitempty"`
	TopLogprobs          json.RawMessage   `json:"top_logprobs,omitempty"`
	TopP                 json.RawMessage   `json:"top_p,omitempty"`
	Truncation           json.RawMessage   `json:"truncation,omitempty"`
	User                 json.RawMessage   `json:"user,omitempty"`
}

type responsesUsageWire struct {
	InputTokens        int64           `json:"input_tokens"`
	OutputTokens       int64           `json:"output_tokens"`
	TotalTokens        int64           `json:"total_tokens"`
	ComputeUnits       json.RawMessage `json:"compute_units,omitempty"`
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
	var diagnostics llmprotocol.Diagnostics
	appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
		"background": len(wire.Background) > 0, "completed_at": len(wire.CompletedAt) > 0,
		"conversation": len(wire.Conversation) > 0, "conversation_id": wire.ConversationID != "",
		"instructions": len(wire.Instructions) > 0, "max_output_tokens": len(wire.MaxOutputTokens) > 0,
		"max_tool_calls": len(wire.MaxToolCalls) > 0, "metadata": len(wire.Metadata) > 0,
		"moderation": len(wire.Moderation) > 0, "output_text": len(wire.OutputText) > 0,
		"parallel_tool_calls":  len(wire.ParallelToolCalls) > 0,
		"previous_response_id": wire.PreviousResponseID != "", "reasoning": len(wire.Reasoning) > 0,
		"prompt": len(wire.Prompt) > 0, "prompt_cache_key": len(wire.PromptCacheKey) > 0,
		"prompt_cache_options":   len(wire.PromptCacheOptions) > 0,
		"prompt_cache_retention": len(wire.PromptCacheRetention) > 0,
		"safety_identifier":      len(wire.SafetyIdentifier) > 0,
		"service_tier":           len(wire.ServiceTier) > 0,
		"store":                  len(wire.Store) > 0, "temperature": len(wire.Temperature) > 0,
		"text": len(wire.Text) > 0, "tool_choice": len(wire.ToolChoice) > 0,
		"tools": len(wire.Tools) > 0, "top_logprobs": len(wire.TopLogprobs) > 0,
		"top_p":      len(wire.TopP) > 0,
		"truncation": len(wire.Truncation) > 0, "user": len(wire.User) > 0,
	}, "response request-echo metadata is not model output")
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.CreatedAt > 0 {
		response.CreatedAt = time.Unix(wire.CreatedAt, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Code), Code: wire.Error.Code, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	for index, item := range wire.Output {
		output, err := decodeResponsesOutputItem(item, index, policy, &diagnostics)
		if err != nil {
			return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
		}
		response.Output = append(response.Output, output)
	}
	if wire.Usage != nil {
		response.Usage = decodeResponsesUsage(*wire.Usage)
		appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
			"usage.compute_units": len(wire.Usage.ComputeUnits) > 0,
		}, "compute-unit accounting has no protocol-neutral token representation")
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
	return response, responseEnvelope(llmprotocol.OpenAIResponsesV1, body, response.Generation, wire.Status, policy), diagnostics, nil
}

func decodeResponsesOutputItem(item responsesItemWire, index int, policy llmprotocol.Policy, diagnostics *llmprotocol.Diagnostics) (llmprotocol.OutputItem, error) {
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
		"output.encrypted_content": len(item.EncryptedContent) > 0,
		"output.phase":             len(item.Phase) > 0,
	}, "response item metadata has no protocol-neutral representation")
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
		content, err := decodeResponsesContent(item.Content, policy, false, true)
		if err != nil {
			return llmprotocol.OutputItem{}, err
		}
		if responsesContentFieldPresent(item.Content, "logprobs") {
			appendProviderFieldOmission(diagnostics, policy, llmprotocol.OpenAIResponsesV1, "output.content.logprobs", "token log probabilities have no protocol-neutral representation")
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

func responsesContentFieldPresent(raw json.RawMessage, field string) bool {
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, part := range parts {
		if value, present := part[field]; present && len(value) > 0 {
			return true
		}
	}
	return false
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
		wire := responsesResponseWire{
			ID: response.ID, Object: "response", Model: response.Model, Status: "failed",
			Error: &responsesErrorWire{Code: response.Error.Code, Message: response.Error.Message},
		}
		if !response.CreatedAt.IsZero() {
			wire.CreatedAt = response.CreatedAt.Unix()
		}
		body, err := marshalWire(wire)
		return body, nil, err
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

func (OpenAIResponsesCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	return decodeOpenAITransportError(body, policy)
}

func (OpenAIResponsesCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	return encodeOpenAITransportError(transportError)
}
