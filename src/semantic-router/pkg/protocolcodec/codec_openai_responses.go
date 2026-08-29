package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
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
		llmprotocol.CapabilityReasoningEffort, llmprotocol.CapabilityRequestMetadata,
		llmprotocol.CapabilityRequestStorage, llmprotocol.CapabilityAutomaticStorage,
		llmprotocol.CapabilityConversationState,
	)
}

type responsesRequestWire struct {
	Model                string                      `json:"model"`
	Input                json.RawMessage             `json:"input"`
	Instructions         json.RawMessage             `json:"instructions,omitempty"`
	Tools                json.RawMessage             `json:"tools,omitempty"`
	ToolChoice           json.RawMessage             `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                       `json:"parallel_tool_calls,omitempty"`
	Temperature          *float64                    `json:"temperature,omitempty"`
	TopP                 *float64                    `json:"top_p,omitempty"`
	MaxOutputTokens      *int64                      `json:"max_output_tokens,omitempty"`
	Metadata             map[string]string           `json:"metadata,omitempty"`
	Text                 *responsesTextWire          `json:"text,omitempty"`
	Stream               bool                        `json:"stream,omitempty"`
	Store                *bool                       `json:"store,omitempty"`
	PreviousResponseID   string                      `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage             `json:"conversation,omitempty"`
	AutoStore            *bool                       `json:"auto_store,omitempty"`
	Reasoning            *responsesReasoningWire     `json:"reasoning,omitempty"`
	Truncation           string                      `json:"truncation,omitempty"`
	User                 string                      `json:"user,omitempty"`
	Background           json.RawMessage             `json:"background,omitempty"`
	ContextManagement    json.RawMessage             `json:"context_management,omitempty"`
	Include              json.RawMessage             `json:"include,omitempty"`
	MaxToolCalls         json.RawMessage             `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage             `json:"moderation,omitempty"`
	Prompt               json.RawMessage             `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage             `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention json.RawMessage             `json:"prompt_cache_retention,omitempty"`
	PromptCacheOptions   json.RawMessage             `json:"prompt_cache_options,omitempty"`
	SafetyIdentifier     json.RawMessage             `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage             `json:"service_tier,omitempty"`
	StreamOptions        *responsesStreamOptionsWire `json:"stream_options,omitempty"`
	TopLogprobs          json.RawMessage             `json:"top_logprobs,omitempty"`
}

type responsesReasoningWire struct {
	Mode            json.RawMessage `json:"mode,omitempty"`
	Effort          string          `json:"effort,omitempty"`
	Summary         json.RawMessage `json:"summary,omitempty"`
	Context         json.RawMessage `json:"context,omitempty"`
	GenerateSummary json.RawMessage `json:"generate_summary,omitempty"`
}

type responsesStreamOptionsWire struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

type responsesTextWire struct {
	Format    responsesFormatWire `json:"format,omitempty"`
	Verbosity json.RawMessage     `json:"verbosity,omitempty"`
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
	Caller           json.RawMessage `json:"caller,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
	Arguments        string          `json:"arguments,omitempty"`
	Output           json.RawMessage `json:"output,omitempty"`
	Summary          json.RawMessage `json:"summary,omitempty"`
	Phase            json.RawMessage `json:"phase,omitempty"`
	EncryptedContent json.RawMessage `json:"encrypted_content,omitempty"`
}

func (wire responsesItemWire) MarshalJSON() ([]byte, error) {
	type itemAlias responsesItemWire
	body, err := json.Marshal(itemAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "message":
		if len(wire.Content) == 0 {
			object["content"] = json.RawMessage(`[]`)
		}
	case "function_call":
		arguments, err := json.Marshal(wire.Arguments)
		if err != nil {
			return nil, err
		}
		object["arguments"] = arguments
	case "reasoning":
		if len(wire.Summary) == 0 {
			object["summary"] = json.RawMessage(`[]`)
		}
	}
	return json.Marshal(object)
}

type responsesContentWire struct {
	Type                  string                     `json:"type"`
	Text                  string                     `json:"text,omitempty"`
	Refusal               string                     `json:"refusal,omitempty"`
	Annotations           *[]responsesAnnotationWire `json:"annotations,omitempty"`
	ImageURL              string                     `json:"image_url,omitempty"`
	FileURL               string                     `json:"file_url,omitempty"`
	FileID                string                     `json:"file_id,omitempty"`
	FileData              string                     `json:"file_data,omitempty"`
	Filename              string                     `json:"filename,omitempty"`
	Detail                string                     `json:"detail,omitempty"`
	Logprobs              json.RawMessage            `json:"logprobs,omitempty"`
	PromptCacheBreakpoint json.RawMessage            `json:"prompt_cache_breakpoint,omitempty"`
}

func (wire responsesContentWire) MarshalJSON() ([]byte, error) {
	type contentAlias responsesContentWire
	body, err := json.Marshal(contentAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "input_text", "output_text", "reasoning_text", "summary_text":
		object["text"], _ = json.Marshal(wire.Text)
	case "refusal":
		object["refusal"], _ = json.Marshal(wire.Refusal)
	}
	if wire.Type == "output_text" && wire.Annotations == nil {
		object["annotations"] = json.RawMessage(`[]`)
	}
	return json.Marshal(object)
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
		"service_tier": wire.ServiceTier,
		"top_logprobs": wire.TopLogprobs,
	}); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if wire.MaxOutputTokens != nil && *wire.MaxOutputTokens < 16 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_responses_max_output_tokens",
			"Responses output token limit must be at least 16",
			nil,
		)
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
	if err := decodeResponsesTools(wire.Tools, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesRequestOptions(wire, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.OpenAIResponsesV1, body, request.Generation, policy), nil, nil
}

func decodeResponsesBaseRequest(wire responsesRequestWire, conversationID string) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		EndUserID: wire.User, PreviousResponseID: wire.PreviousResponseID, ConversationID: conversationID,
		Truncation: wire.Truncation,
		Store:      wire.Store, AutoStore: wire.AutoStore, ParallelToolCalls: wire.ParallelToolCalls,
		Sampling: llmprotocol.Sampling{Temperature: wire.Temperature, TopP: wire.TopP, MaxOutputTokens: wire.MaxOutputTokens},
		Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIResponsesV1},
	}
	if wire.StreamOptions != nil {
		request.StreamOptions.IncludeObfuscation = wire.StreamOptions.IncludeObfuscation
	}
	return request
}

func decodeResponsesReasoningRequest(reasoning *responsesReasoningWire, request *llmprotocol.Request) error {
	if reasoning == nil {
		return nil
	}
	request.ReasoningEffort = reasoning.Effort
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"reasoning.mode":             reasoning.Mode,
		"reasoning.summary":          reasoning.Summary,
		"reasoning.context":          reasoning.Context,
		"reasoning.generate_summary": reasoning.GenerateSummary,
	}); err != nil {
		return err
	}
	return nil
}

func decodeResponsesInstructions(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var text string
	if err := decodeWireValue(raw, &text, policy); err != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_instructions",
			"Responses instructions must be a string",
			err,
		)
	}
	request.Instructions = []llmprotocol.InstructionBlock{{
		Role:    llmprotocol.RoleDeveloper,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}}
	return nil
}

func decodeResponsesTools(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var toolBodies []json.RawMessage
	if err := decodeWireValue(raw, &toolBodies, policy); err != nil {
		return err
	}
	for _, toolBody := range toolBodies {
		var discriminator struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(toolBody, &discriminator); err != nil || discriminator.Type == "" {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses tool type is required", err)
		}
		if discriminator.Type != "function" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools enter the model protocol", nil)
		}
		var tool responsesToolWire
		if err := decodeWireValue(toolBody, &tool, policy); err != nil {
			return err
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

func decodeResponsesRequestOptions(wire responsesRequestWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	choice, err := decodeResponsesToolChoice(wire.ToolChoice, policy)
	if err != nil {
		return err
	}
	request.ToolChoice = choice
	if wire.Text != nil {
		if err := rejectUnsupportedRequestField("text.verbosity", wire.Text.Verbosity); err != nil {
			return err
		}
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
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "input_required", "input is required", nil)
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}})
		return nil
	}
	var itemBodies []json.RawMessage
	if err := decodeWireValue(raw, &itemBodies, policy); err != nil {
		return err
	}
	for index, itemBody := range itemBodies {
		item, err := decodeResponsesItemWire(itemBody, policy, false)
		if err != nil {
			return err
		}
		if err := decodeResponsesInputItem(item, index, request, policy); err != nil {
			return err
		}
	}
	return nil
}

var responsesItemUnionFields = []string{
	"arguments", "call_id", "caller", "content", "encrypted_content", "id", "name", "namespace",
	"output", "phase", "role", "status", "summary", "type",
}

func decodeResponsesItemWire(body json.RawMessage, policy llmprotocol.Policy, providerOutput bool) (responsesItemWire, error) {
	var discriminator struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(body, &discriminator); err != nil {
		return responsesItemWire{}, responsesItemDecodeError(providerOutput, "item discriminator is invalid", err)
	}
	itemType := discriminator.Type
	if itemType == "" && !providerOutput {
		itemType = "message"
	}
	if !isSupportedResponsesItemType(itemType, providerOutput) {
		code := "unsupported_input_item"
		message := "Responses input item is unsupported"
		if providerOutput {
			code = "unsupported_output_item"
			message = "Responses output item is unsupported"
		}
		return responsesItemWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, code, message, nil)
	}
	var item responsesItemWire
	var err error
	if providerOutput {
		err = decodeProviderValue(body, &item, policy)
	} else {
		err = decodeWireValue(body, &item, policy)
	}
	if err != nil {
		return responsesItemWire{}, err
	}
	if err := validateResponsesItemVariant(body, itemType, providerOutput); err != nil {
		return responsesItemWire{}, err
	}
	return item, nil
}

func isSupportedResponsesItemType(itemType string, providerOutput bool) bool {
	if providerOutput {
		return itemType == "message" || itemType == "function_call" || itemType == "reasoning"
	}
	switch itemType {
	case "message", "function_call", "function_call_output", "reasoning", "item_reference":
		return true
	default:
		return false
	}
}

func validateResponsesItemVariant(body json.RawMessage, itemType string, providerOutput bool) error {
	allowed := map[string]struct{}{}
	for _, field := range responsesItemAllowedFields(itemType) {
		allowed[field] = struct{}{}
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return responsesItemDecodeError(providerOutput, "item object is invalid", err)
	}
	for _, field := range responsesItemUnionFields {
		if _, present := object[field]; !present {
			continue
		}
		if _, valid := allowed[field]; valid {
			continue
		}
		return responsesItemDecodeError(providerOutput, "item includes a field from another union variant: "+field, nil)
	}
	return nil
}

func responsesItemAllowedFields(itemType string) []string {
	switch itemType {
	case "message":
		return []string{"content", "id", "phase", "role", "status", "type"}
	case "function_call":
		return []string{"arguments", "call_id", "caller", "id", "name", "namespace", "status", "type"}
	case "function_call_output":
		return []string{"call_id", "caller", "id", "name", "namespace", "output", "status", "type"}
	case "reasoning":
		return []string{"content", "encrypted_content", "id", "status", "summary", "type"}
	case "item_reference":
		return []string{"id", "type"}
	default:
		return nil
	}
}

func responsesItemDecodeError(providerOutput bool, detail string, cause error) error {
	if providerOutput {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_response_item",
			"Responses upstream response "+detail,
			cause,
		)
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorInvalidRequest,
		"invalid_input_item_variant",
		"Responses input "+detail,
		cause,
	)
}

func decodeResponsesInputItem(
	item responsesItemWire,
	index int,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"input.caller":            item.Caller,
		"input.encrypted_content": item.EncryptedContent,
		"input.phase":             item.Phase,
	}); err != nil {
		return err
	}
	if item.Namespace != "" {
		return rejectUnsupportedRequestField("input.namespace", json.RawMessage(`true`))
	}
	if item.Status != "" {
		return rejectUnsupportedRequestField("input.status", json.RawMessage(`true`))
	}
	switch item.Type {
	case "", "message":
		return decodeResponsesMessageItem(item, request, policy)
	case "function_call":
		request.Messages = append(request.Messages, decodeResponsesFunctionCall(item, index, policy))
		return nil
	case "function_call_output":
		if item.Name != "" {
			return rejectUnsupportedRequestField("input.function_call_output.name", json.RawMessage(`true`))
		}
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
	if role == llmprotocol.RoleTool {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_role",
			"Responses tool results must use function_call_output items",
			nil,
		)
	}
	context := responsesInputMessageContent
	if role == llmprotocol.RoleAssistant {
		context = responsesAssistantHistoryContent
	}
	content, err := decodeResponsesContent(item.Content, policy, context)
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
	return llmprotocol.Message{ID: item.ID, Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind:     llmprotocol.ContentToolCall,
		ToolCall: &llmprotocol.ToolCall{ID: id, Name: item.Name, Arguments: item.Arguments},
	}}}
}

func decodeResponsesFunctionResult(item responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	content, err := decodeResponsesContent(item.Output, policy, responsesFunctionOutputContent)
	if err != nil {
		return err
	}
	request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind:       llmprotocol.ContentToolResult,
		ToolResult: &llmprotocol.ToolResult{CallID: item.CallID, Content: content},
	}}})
	return nil
}

func decodeResponsesReasoningItem(item responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	content, err := decodeResponsesReasoning(item.Summary, policy, false)
	if err != nil {
		return err
	}
	reasoning, err := decodeResponsesContent(item.Content, policy, responsesRequestReasoningContent)
	if err != nil {
		return err
	}
	content = append(content, reasoning...)
	request.Messages = append(request.Messages, llmprotocol.Message{ID: item.ID, Role: llmprotocol.RoleAssistant, Content: content})
	return nil
}

type responsesContentContext uint8

const (
	responsesInputMessageContent responsesContentContext = iota
	responsesAssistantHistoryContent
	responsesFunctionOutputContent
	responsesProviderOutputContent
	responsesRequestReasoningContent
	responsesProviderReasoningContent
)

func decodeResponsesContent(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	context responsesContentContext,
) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return decodeResponsesStringContent(text, context)
	}
	partBodies, err := decodeResponsesContentBodies(raw, policy, context)
	if err != nil {
		return nil, err
	}
	return decodeResponsesContentParts(partBodies, policy, context)
}

func decodeResponsesStringContent(text string, context responsesContentContext) ([]llmprotocol.Content, error) {
	if isResponsesProviderContent(context) {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_response_content",
			"Responses provider output content must be an array",
			nil,
		)
	}
	if context == responsesRequestReasoningContent {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_reasoning_content",
			"Responses reasoning content must be an array",
			nil,
		)
	}
	return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
}

func decodeResponsesContentBodies(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	context responsesContentContext,
) ([]json.RawMessage, error) {
	var partBodies []json.RawMessage
	if isResponsesProviderContent(context) {
		return partBodies, decodeProviderValue(raw, &partBodies, policy)
	}
	return partBodies, decodeWireValue(raw, &partBodies, policy)
}

func decodeResponsesContentParts(
	partBodies []json.RawMessage,
	policy llmprotocol.Policy,
	context responsesContentContext,
) ([]llmprotocol.Content, error) {
	result := make([]llmprotocol.Content, 0, len(partBodies))
	var assistantFamily string
	for _, partBody := range partBodies {
		family, content, err := decodeResponsesContentPart(partBody, context, policy)
		if err != nil {
			return nil, err
		}
		if err := trackResponsesAssistantFamily(context, family, &assistantFamily); err != nil {
			return nil, err
		}
		result = append(result, content)
	}
	return result, nil
}

func trackResponsesAssistantFamily(context responsesContentContext, family string, current *string) error {
	if context != responsesAssistantHistoryContent {
		return nil
	}
	if *current == "" {
		*current = family
		return nil
	}
	if *current == family {
		return nil
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorInvalidRequest,
		"mixed_assistant_content",
		"Responses assistant history must use one official content union",
		nil,
	)
}

func isResponsesProviderContent(context responsesContentContext) bool {
	return context == responsesProviderOutputContent || context == responsesProviderReasoningContent
}

func decodeResponsesContentPart(
	body json.RawMessage,
	context responsesContentContext,
	policy llmprotocol.Policy,
) (string, llmprotocol.Content, error) {
	var part responsesContentWire
	if err := decodeResponsesContentWire(body, &part, policy, context); err != nil {
		return "", llmprotocol.Content{}, err
	}
	if err := validateResponsesContentVariant(body, part.Type, context); err != nil {
		return "", llmprotocol.Content{}, err
	}
	unsupported := map[string]json.RawMessage{
		"content.prompt_cache_breakpoint": part.PromptCacheBreakpoint,
	}
	if context != responsesProviderOutputContent {
		unsupported["content.logprobs"] = part.Logprobs
	}
	if err := rejectUnsupportedRequestFields(unsupported); err != nil {
		return "", llmprotocol.Content{}, err
	}
	return decodeResponsesTypedContent(part, context)
}

func decodeResponsesContentWire(
	body json.RawMessage,
	part *responsesContentWire,
	policy llmprotocol.Policy,
	context responsesContentContext,
) error {
	if isResponsesProviderContent(context) {
		return decodeProviderValue(body, part, policy)
	}
	return decodeWireValue(body, part, policy)
}

func decodeResponsesTypedContent(part responsesContentWire, context responsesContentContext) (string, llmprotocol.Content, error) {
	switch part.Type {
	case "input_text", "input_image", "input_file":
		return decodeResponsesInputContent(part, context)
	case "output_text", "refusal":
		return decodeResponsesOutputContent(part, context)
	case "reasoning_text":
		return decodeResponsesReasoningContent(part, context)
	default:
		return "", llmprotocol.Content{}, unsupportedResponsesContent(part.Type, context)
	}
}

func decodeResponsesInputContent(part responsesContentWire, context responsesContentContext) (string, llmprotocol.Content, error) {
	if !responsesInputContentAllowed(context) {
		return "", llmprotocol.Content{}, unsupportedResponsesContent(part.Type, context)
	}
	switch part.Type {
	case "input_text":
		return "input", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text}, nil
	case "input_image":
		content, err := decodeResponsesImage(part)
		return "input", content, err
	default:
		return "input", decodeResponsesFile(part), nil
	}
}

func decodeResponsesOutputContent(part responsesContentWire, context responsesContentContext) (string, llmprotocol.Content, error) {
	if context != responsesAssistantHistoryContent && context != responsesProviderOutputContent {
		return "", llmprotocol.Content{}, unsupportedResponsesContent(part.Type, context)
	}
	if part.Type == "refusal" {
		return "output", llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.Refusal}, nil
	}
	citations, err := decodeResponsesAnnotations(responsesAnnotationsValue(part.Annotations))
	return "output", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text, Citations: citations}, err
}

func decodeResponsesReasoningContent(part responsesContentWire, context responsesContentContext) (string, llmprotocol.Content, error) {
	if context != responsesRequestReasoningContent && context != responsesProviderReasoningContent {
		return "", llmprotocol.Content{}, unsupportedResponsesContent(part.Type, context)
	}
	return "reasoning", llmprotocol.Content{
		Kind: llmprotocol.ContentReasoning, Text: part.Text, Reasoning: llmprotocol.ReasoningScopeText,
	}, nil
}

func responsesInputContentAllowed(context responsesContentContext) bool {
	return context == responsesInputMessageContent ||
		context == responsesAssistantHistoryContent ||
		context == responsesFunctionOutputContent
}

func validateResponsesContentVariant(
	body json.RawMessage,
	typeName string,
	context responsesContentContext,
) error {
	allowedByType := map[string][]string{
		"input_text":     {"prompt_cache_breakpoint", "text", "type"},
		"input_image":    {"detail", "file_id", "image_url", "prompt_cache_breakpoint", "type"},
		"input_file":     {"detail", "file_data", "file_id", "file_url", "filename", "prompt_cache_breakpoint", "type"},
		"output_text":    {"annotations", "logprobs", "text", "type"},
		"refusal":        {"refusal", "type"},
		"reasoning_text": {"text", "type"},
	}
	allowed, recognized := allowedByType[typeName]
	if !recognized {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return err
	}
	if err := requireResponsesContentFields(object, typeName, context); err != nil {
		return err
	}
	return rejectResponsesContentVariantFields(object, allowed, context)
}

func requireResponsesContentFields(
	object map[string]json.RawMessage,
	typeName string,
	context responsesContentContext,
) error {
	requiredByType := map[string][]string{
		"input_text":     {"text"},
		"output_text":    {"text"},
		"refusal":        {"refusal"},
		"reasoning_text": {"text"},
	}
	for _, name := range requiredByType[typeName] {
		if _, present := object[name]; present {
			continue
		}
		category := llmprotocol.ErrorInvalidRequest
		code := "invalid_content_variant"
		message := "Responses content is missing the required field: " + name
		if isResponsesProviderContent(context) {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Responses provider output is missing the required field: " + name
		}
		return llmprotocol.NewError(category, code, message, nil)
	}
	return nil
}

func rejectResponsesContentVariantFields(
	object map[string]json.RawMessage,
	allowed []string,
	context responsesContentContext,
) error {
	known := []string{
		"annotations", "detail", "file_data", "file_id", "file_url", "filename", "image_url",
		"logprobs", "prompt_cache_breakpoint", "refusal", "text", "type",
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, name := range allowed {
		allowedSet[name] = struct{}{}
	}
	for _, name := range known {
		if _, present := object[name]; !present {
			continue
		}
		if _, valid := allowedSet[name]; valid {
			continue
		}
		category := llmprotocol.ErrorInvalidRequest
		code := "invalid_content_variant"
		message := "Responses content includes a field from a different union variant"
		if isResponsesProviderContent(context) {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Responses provider output mixes content union variants"
		}
		return llmprotocol.NewError(category, code, message+": "+name, nil)
	}
	return nil
}

func unsupportedResponsesContent(contentType string, context responsesContentContext) error {
	category := llmprotocol.ErrorUnsupportedFeature
	code := "unsupported_content"
	message := "Responses content type is unsupported in this position"
	if context == responsesProviderOutputContent || context == responsesProviderReasoningContent {
		category = llmprotocol.ErrorUpstreamUnavailable
		code = "invalid_response_content"
		message = "Responses provider output contains content in an invalid position"
	}
	return llmprotocol.NewError(category, code, message+": "+contentType, nil)
}

func decodeResponsesImage(part responsesContentWire) (llmprotocol.Content, error) {
	if mediaType, data, inline := decodeDataURL(part.ImageURL); inline {
		return llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: mediaType, Data: data, Detail: part.Detail}, nil
	}
	return llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: part.ImageURL, FileID: part.FileID, Detail: part.Detail}, nil
}

func decodeResponsesFile(part responsesContentWire) llmprotocol.Content {
	content := llmprotocol.Content{Kind: llmprotocol.ContentFile, URL: part.FileURL, FileID: part.FileID, Filename: part.Filename, Detail: part.Detail}
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

func responsesAnnotations(citations []llmprotocol.Citation) *[]responsesAnnotationWire {
	annotations := encodeResponsesAnnotations(citations)
	return &annotations
}

func responsesAnnotationsValue(annotations *[]responsesAnnotationWire) []responsesAnnotationWire {
	if annotations == nil {
		return nil
	}
	return *annotations
}

func decodeResponsesToolChoice(raw json.RawMessage, policy llmprotocol.Policy) (llmprotocol.ToolChoice, error) {
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
	var discriminator struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &discriminator) == nil && unsupportedResponsesToolChoice(discriminator.Type) {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_tool_choice",
			"Responses tool choice cannot be represented by the neutral protocol",
			nil,
		)
	}
	var named struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if decodeWireValue(raw, &named, policy) != nil || named.Type != "function" || named.Name == "" {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Responses tool choice is invalid", nil)
	}
	return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: named.Name}, nil
}

func unsupportedResponsesToolChoice(typeName string) bool {
	switch typeName {
	case "allowed_tools", "apply_patch", "code_interpreter", "computer", "computer_use",
		"computer_use_preview", "custom", "file_search", "image_generation", "mcp",
		"programmatic_tool_calling", "shell", "web_search_preview", "web_search_preview_2025_03_11":
		return true
	default:
		return false
	}
}

func (OpenAIResponsesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	if err := validateResponsesEncodableRequest(request); err != nil {
		return nil, nil, err
	}
	wire, err := encodeResponsesRequestWire(request)
	if err != nil {
		return nil, nil, err
	}
	body, err := marshalWire(wire)
	return body, nil, err
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
	wire.Tools = encodeResponsesTools(request.Tools)
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

func encodeResponsesTools(input []llmprotocol.Tool) json.RawMessage {
	if len(input) == 0 {
		return nil
	}
	tools := make([]responsesToolWire, 0, len(input))
	for _, tool := range input {
		tools = append(tools, responsesToolWire{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict})
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
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		if err := state.flushReasoning(); err != nil {
			return err
		}
		return state.appendToolCall(content.ToolCall)
	case llmprotocol.ContentToolResult:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		if err := state.flushReasoning(); err != nil {
			return err
		}
		return state.appendToolResult(content.ToolResult)
	case llmprotocol.ContentReasoning:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		state.reasoning = append(state.reasoning, content)
	default:
		if err := state.flushReasoning(); err != nil {
			return err
		}
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
	default:
		return nil
	}
}

type responsesResponseWire struct {
	ID                string              `json:"id"`
	Object            string              `json:"object,omitempty"`
	CreatedAt         int64               `json:"created_at"`
	Model             string              `json:"model"`
	Status            string              `json:"status,omitempty"`
	Output            json.RawMessage     `json:"output"`
	Usage             *responsesUsageWire `json:"usage,omitempty"`
	Error             *responsesErrorWire `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	PreviousResponseID   string            `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage   `json:"conversation,omitempty"`
	ConversationID       string            `json:"conversation_id,omitempty"`
	Metadata             map[string]string `json:"metadata"`
	Background           json.RawMessage   `json:"background,omitempty"`
	CompletedAt          json.RawMessage   `json:"completed_at,omitempty"`
	Instructions         json.RawMessage   `json:"instructions"`
	MaxOutputTokens      json.RawMessage   `json:"max_output_tokens,omitempty"`
	MaxToolCalls         json.RawMessage   `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage   `json:"moderation,omitempty"`
	OutputText           json.RawMessage   `json:"output_text,omitempty"`
	ParallelToolCalls    json.RawMessage   `json:"parallel_tool_calls"`
	Prompt               json.RawMessage   `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage   `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   json.RawMessage   `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention json.RawMessage   `json:"prompt_cache_retention,omitempty"`
	Reasoning            json.RawMessage   `json:"reasoning,omitempty"`
	SafetyIdentifier     json.RawMessage   `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage   `json:"service_tier,omitempty"`
	Store                json.RawMessage   `json:"store,omitempty"`
	Temperature          json.RawMessage   `json:"temperature"`
	Text                 json.RawMessage   `json:"text,omitempty"`
	ToolChoice           json.RawMessage   `json:"tool_choice"`
	Tools                json.RawMessage   `json:"tools"`
	TopLogprobs          json.RawMessage   `json:"top_logprobs,omitempty"`
	TopP                 json.RawMessage   `json:"top_p"`
	Truncation           json.RawMessage   `json:"truncation,omitempty"`
	User                 json.RawMessage   `json:"user,omitempty"`
}

func newResponsesResponseWire(id, model, status string, createdAt int64, previousResponseID string) responsesResponseWire {
	return responsesResponseWire{
		ID: id, Object: "response", CreatedAt: createdAt, Model: model, Status: status,
		Output:             json.RawMessage(`[]`),
		Metadata:           map[string]string{},
		Instructions:       json.RawMessage(`null`),
		ParallelToolCalls:  json.RawMessage(`true`),
		Temperature:        json.RawMessage(`null`),
		ToolChoice:         json.RawMessage(`"auto"`),
		Tools:              json.RawMessage(`[]`),
		TopP:               json.RawMessage(`null`),
		PreviousResponseID: previousResponseID,
	}
}

type responsesUsageWire struct {
	InputTokens         int64                        `json:"input_tokens"`
	OutputTokens        int64                        `json:"output_tokens"`
	TotalTokens         int64                        `json:"total_tokens"`
	ComputeUnits        json.RawMessage              `json:"compute_units,omitempty"`
	InputTokensDetails  *responsesInputUsageDetails  `json:"input_tokens_details"`
	OutputTokensDetails *responsesOutputUsageDetails `json:"output_tokens_details"`
}

type responsesInputUsageDetails struct {
	CachedTokens     int64 `json:"cached_tokens"`
	CacheWriteTokens int64 `json:"cache_write_tokens"`
}

type responsesOutputUsageDetails struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

type responsesErrorWire struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func responsesErrorCode(protocolError *llmprotocol.ProtocolError) string {
	if protocolError.Code != "" {
		return protocolError.Code
	}
	switch protocolError.Category {
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_exceeded"
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_prompt"
	default:
		return "server_error"
	}
}

func (OpenAIResponsesCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire responsesResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateResponsesResponseResource(wire, false); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	diagnostics := responsesResponseMetadataDiagnostics(wire, policy)
	response, err := decodeResponsesResponseResource(wire, policy, &diagnostics)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	return response, responseEnvelope(llmprotocol.OpenAIResponsesV1, body, response.Generation, wire.Status, policy), diagnostics, nil
}

func responsesResponseMetadataDiagnostics(wire responsesResponseWire, policy llmprotocol.Policy) llmprotocol.Diagnostics {
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
	return diagnostics
}

func decodeResponsesResponseResource(
	wire responsesResponseWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) (llmprotocol.Response, error) {
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.CreatedAt > 0 {
		response.CreatedAt = time.Unix(wire.CreatedAt, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Code), Code: wire.Error.Code, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	output, err := decodeResponsesOutput(wire.Output, policy, diagnostics)
	if err != nil {
		return llmprotocol.Response{}, err
	}
	response.Output = output
	if wire.Usage != nil {
		response.Usage = decodeResponsesUsage(*wire.Usage)
		appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
			"usage.compute_units": len(wire.Usage.ComputeUnits) > 0,
		}, "compute-unit accounting has no protocol-neutral token representation")
	}
	if response.Error == nil {
		response.StopReason = llmprotocol.StopEndTurn
	}
	response.SourceStopReason = wire.Status
	response.StopReason = decodeResponsesStopReason(wire, response.StopReason)
	return response, nil
}

func decodeResponsesStopReason(wire responsesResponseWire, fallback llmprotocol.StopReason) llmprotocol.StopReason {
	if wire.Status == "incomplete" && wire.IncompleteDetails != nil {
		switch wire.IncompleteDetails.Reason {
		case "max_output_tokens":
			return llmprotocol.StopMaxTokens
		case "content_filter":
			return llmprotocol.StopContentFilter
		default:
			return llmprotocol.StopUnknown
		}
	}
	switch wire.Status {
	case "failed":
		return llmprotocol.StopError
	case "cancelled":
		return llmprotocol.StopCanceled
	default:
		return fallback
	}
}

func decodeResponsesOutput(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) ([]llmprotocol.OutputItem, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var itemBodies []json.RawMessage
	if err := decodeProviderValue(raw, &itemBodies, policy); err != nil {
		return nil, err
	}
	output := make([]llmprotocol.OutputItem, 0, len(itemBodies))
	for index, itemBody := range itemBodies {
		item, err := decodeResponsesItemWire(itemBody, policy, true)
		if err != nil {
			return nil, err
		}
		if err := validateResponsesOutputItemResource(itemBody, item); err != nil {
			return nil, err
		}
		decoded, err := decodeResponsesOutputItem(item, index, policy, diagnostics)
		if err != nil {
			return nil, err
		}
		output = append(output, decoded)
	}
	return output, nil
}

func decodeResponsesOutputItem(item responsesItemWire, index int, policy llmprotocol.Policy, diagnostics *llmprotocol.Diagnostics) (llmprotocol.OutputItem, error) {
	if err := validateResponsesOutputItemStatus(item); err != nil {
		return llmprotocol.OutputItem{}, err
	}
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
		"output.caller":            len(item.Caller) > 0,
		"output.encrypted_content": len(item.EncryptedContent) > 0,
		"output.namespace":         item.Namespace != "",
		"output.phase":             len(item.Phase) > 0,
		"output.status":            item.Status != "",
	}, "response item metadata has no protocol-neutral representation")
	id := item.ID
	if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		id = llmprotocol.StableID("response-output", fmt.Sprint(index), item.Type, item.CallID)
	}
	output := llmprotocol.OutputItem{ID: id, Role: llmprotocol.RoleAssistant}
	switch item.Type {
	case "message":
		return decodeResponsesMessageOutput(output, item, policy, diagnostics)
	case "function_call":
		output.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}}}
	case "reasoning":
		return decodeResponsesReasoningOutput(output, item, policy)
	default:
		return llmprotocol.OutputItem{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses output item is unsupported", nil)
	}
	return output, nil
}

func decodeResponsesMessageOutput(
	output llmprotocol.OutputItem,
	item responsesItemWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) (llmprotocol.OutputItem, error) {
	role, err := canonicalRole(item.Role)
	if err != nil {
		return llmprotocol.OutputItem{}, invalidProviderResponse("invalid_response_role", "Responses output messages must use the assistant role")
	}
	if role != llmprotocol.RoleAssistant {
		return llmprotocol.OutputItem{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_response_role",
			"Responses output messages must use the assistant role",
			nil,
		)
	}
	content, err := decodeResponsesContent(item.Content, policy, responsesProviderOutputContent)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	if responsesContentFieldPresent(item.Content, "logprobs") {
		appendProviderFieldOmission(diagnostics, policy, llmprotocol.OpenAIResponsesV1, "output.content.logprobs", "token log probabilities have no protocol-neutral representation")
	}
	output.Role, output.Content = role, content
	return output, nil
}

func decodeResponsesReasoningOutput(
	output llmprotocol.OutputItem,
	item responsesItemWire,
	policy llmprotocol.Policy,
) (llmprotocol.OutputItem, error) {
	content, err := decodeResponsesReasoning(item.Summary, policy, true)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	reasoning, err := decodeResponsesContent(item.Content, policy, responsesProviderReasoningContent)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	output.Content = append(content, reasoning...)
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
	cached, cacheWrite := int64(0), int64(0)
	if wire.InputTokensDetails != nil {
		cached = wire.InputTokensDetails.CachedTokens
		cacheWrite = wire.InputTokensDetails.CacheWriteTokens
	}
	reasoning := int64(0)
	if wire.OutputTokensDetails != nil {
		reasoning = wire.OutputTokensDetails.ReasoningTokens
	}
	uncached := int64(-1)
	if cached >= 0 && cacheWrite >= 0 && wire.InputTokens >= cached && cacheWrite <= wire.InputTokens-cached {
		uncached = wire.InputTokens - cached - cacheWrite
	}
	other := int64(-1)
	if reasoning >= 0 && wire.OutputTokens >= reasoning {
		other = wire.OutputTokens - reasoning
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(uncached), InputCacheRead: authoritative(cached), InputCacheWrite: authoritative(cacheWrite),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(wire.InputTokens), OutputTotal: authoritative(wire.OutputTokens), Total: authoritative(wire.TotalTokens),
	}
}

func (OpenAIResponsesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	if response.Error != nil {
		body, err := encodeResponsesErrorResource(response, envelope)
		return body, nil, err
	}
	var diagnostics llmprotocol.Diagnostics
	if len(response.Alternatives) > 0 {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIResponsesV1, "response.alternatives", "Responses has one primary output sequence"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire, err := encodeResponsesSuccessResource(response, envelope)
	if err != nil {
		return nil, diagnostics, err
	}
	wire.OutputText = encodeResponsesOutputText(response.Output)
	wire.Usage = encodeResponsesUsage(response.Usage)
	if err := applyResponsesStopReason(&wire, response.StopReason, envelope.Format, policy, &diagnostics); err != nil {
		return nil, diagnostics, err
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeResponsesErrorResource(response llmprotocol.Response, envelope llmprotocol.Envelope) ([]byte, error) {
	wire := newResponsesResponseWire(
		response.ID,
		response.Model,
		"failed",
		responsesCreatedAt(response.CreatedAt),
		envelope.ResponseRender.PreviousResponseID,
	)
	wire.Error = &responsesErrorWire{Code: responsesErrorCode(response.Error), Message: response.Error.Message}
	if response.Usage.State == llmprotocol.UsageAvailable {
		wire.Usage = encodeResponsesUsage(response.Usage)
	}
	return marshalWire(wire)
}

func encodeResponsesSuccessResource(response llmprotocol.Response, envelope llmprotocol.Envelope) (responsesResponseWire, error) {
	wire := newResponsesResponseWire(
		response.ID,
		response.Model,
		"completed",
		responsesCreatedAt(response.CreatedAt),
		envelope.ResponseRender.PreviousResponseID,
	)
	items := make([]responsesItemWire, 0, len(response.Output))
	for _, item := range response.Output {
		encoded, err := encodeResponsesOutputItem(item)
		if err != nil {
			return responsesResponseWire{}, err
		}
		items = append(items, encoded...)
	}
	if len(items) == 0 {
		return wire, nil
	}
	encoded, err := json.Marshal(items)
	wire.Output = encoded
	return wire, err
}

func responsesCreatedAt(createdAt time.Time) int64 {
	if createdAt.IsZero() {
		return 0
	}
	return createdAt.Unix()
}

func applyResponsesStopReason(
	wire *responsesResponseWire,
	stopReason llmprotocol.StopReason,
	source llmprotocol.WireFormat,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) error {
	switch stopReason {
	case llmprotocol.StopMaxTokens, llmprotocol.StopContentFilter:
		wire.Status = "incomplete"
		reason := "max_output_tokens"
		if stopReason == llmprotocol.StopContentFilter {
			reason = "content_filter"
		}
		wire.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: reason}
		return nil
	case llmprotocol.StopPaused, llmprotocol.StopContextWindow, llmprotocol.StopCanceled, llmprotocol.StopUnknown:
		return appendLossy(diagnostics, policy, source, llmprotocol.OpenAIResponsesV1, "response.stop_reason", "Responses cannot represent the source terminal reason")
	default:
		return nil
	}
}

func encodeResponsesOutputText(items []llmprotocol.OutputItem) json.RawMessage {
	var text strings.Builder
	for _, item := range items {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentText {
				text.WriteString(content.Text)
			}
		}
	}
	if text.Len() == 0 {
		return nil
	}
	encoded, _ := json.Marshal(text.String())
	return encoded
}

func encodeResponsesOutputItem(item llmprotocol.OutputItem) ([]responsesItemWire, error) {
	message := llmprotocol.Message(item)
	return encodeResponsesMessage(message, "output")
}

func encodeResponsesUsage(usage llmprotocol.Usage) *responsesUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	wire := &responsesUsageWire{
		InputTokens:  tokenValue(usage.InputTotal),
		OutputTokens: tokenValue(usage.OutputTotal),
		TotalTokens:  tokenValue(usage.Total),
		InputTokensDetails: &responsesInputUsageDetails{
			CachedTokens: tokenValue(usage.InputCacheRead), CacheWriteTokens: tokenValue(usage.InputCacheWrite),
		},
		OutputTokensDetails: &responsesOutputUsageDetails{ReasoningTokens: tokenValue(usage.OutputReasoning)},
	}
	if wire.TotalTokens == 0 {
		wire.TotalTokens = wire.InputTokens + wire.OutputTokens
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
