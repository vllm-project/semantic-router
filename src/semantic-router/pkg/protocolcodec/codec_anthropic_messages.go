package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type AnthropicMessagesCodec struct{}

func (AnthropicMessagesCodec) Format() llmprotocol.WireFormat { return llmprotocol.AnthropicMessagesV1 }
func (AnthropicMessagesCodec) Stateless() bool                { return true }
func (AnthropicMessagesCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
	)
}

type anthropicRequestWire struct {
	Model         string                   `json:"model"`
	System        json.RawMessage          `json:"system,omitempty"`
	Messages      []anthropicMessageWire   `json:"messages"`
	MaxTokens     int64                    `json:"max_tokens"`
	Temperature   *float64                 `json:"temperature,omitempty"`
	TopP          *float64                 `json:"top_p,omitempty"`
	TopK          *int64                   `json:"top_k,omitempty"`
	StopSequences []string                 `json:"stop_sequences,omitempty"`
	Tools         []anthropicToolWire      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoiceWire `json:"tool_choice,omitempty"`
	Metadata      *struct {
		UserID string `json:"user_id,omitempty"`
	} `json:"metadata,omitempty"`
	Thinking     *anthropicThinkingWire `json:"thinking,omitempty"`
	Stream       bool                   `json:"stream,omitempty"`
	InferenceGeo json.RawMessage        `json:"inference_geo,omitempty"`
	Container    json.RawMessage        `json:"container,omitempty"`
	CacheControl json.RawMessage        `json:"cache_control,omitempty"`
	OutputConfig json.RawMessage        `json:"output_config,omitempty"`
	ServiceTier  json.RawMessage        `json:"service_tier,omitempty"`
}

type anthropicMessageWire struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicContentWire struct {
	Type            string                    `json:"type"`
	Text            string                    `json:"text,omitempty"`
	Thinking        string                    `json:"thinking,omitempty"`
	Signature       string                    `json:"signature,omitempty"`
	ID              string                    `json:"id,omitempty"`
	Name            string                    `json:"name,omitempty"`
	Input           json.RawMessage           `json:"input,omitempty"`
	ToolUseID       string                    `json:"tool_use_id,omitempty"`
	Content         json.RawMessage           `json:"content,omitempty"`
	IsError         *bool                     `json:"is_error,omitempty"`
	Source          *anthropicMediaSourceWire `json:"source,omitempty"`
	Citations       json.RawMessage           `json:"citations,omitempty"`
	CacheControl    json.RawMessage           `json:"cache_control,omitempty"`
	Caller          json.RawMessage           `json:"caller,omitempty"`
	Context         json.RawMessage           `json:"context,omitempty"`
	Title           json.RawMessage           `json:"title,omitempty"`
	ToolsetName     json.RawMessage           `json:"toolset_name,omitempty"`
	Transformations json.RawMessage           `json:"transformations,omitempty"`
}

type anthropicMediaSourceWire struct {
	Type      string          `json:"type"`
	MediaType string          `json:"media_type,omitempty"`
	Data      string          `json:"data,omitempty"`
	URL       string          `json:"url,omitempty"`
	FileID    string          `json:"file_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

type anthropicToolWire struct {
	Name                string          `json:"name"`
	Description         string          `json:"description,omitempty"`
	InputSchema         json.RawMessage `json:"input_schema"`
	Strict              *bool           `json:"strict,omitempty"`
	Type                string          `json:"type,omitempty"`
	AllowedCallers      json.RawMessage `json:"allowed_callers,omitempty"`
	CacheControl        json.RawMessage `json:"cache_control,omitempty"`
	DeferLoading        json.RawMessage `json:"defer_loading,omitempty"`
	EagerInputStreaming json.RawMessage `json:"eager_input_streaming,omitempty"`
	InputExamples       json.RawMessage `json:"input_examples,omitempty"`
}

type anthropicToolChoiceWire struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type anthropicThinkingWire struct {
	Type         string `json:"type"`
	BudgetTokens int64  `json:"budget_tokens,omitempty"`
}

func (AnthropicMessagesCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire anthropicRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"inference_geo": wire.InferenceGeo, "container": wire.Container,
		"cache_control": wire.CacheControl, "output_config": wire.OutputConfig,
		"service_tier": wire.ServiceTier,
	}); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request := decodeAnthropicBaseRequest(wire)
	if err := decodeAnthropicThinking(wire.Thinking, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeAnthropicSystem(wire.System, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeAnthropicMessages(wire.Messages, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeAnthropicTools(wire.Tools, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeAnthropicToolChoice(wire.ToolChoice, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.AnthropicMessagesV1, body, request.Generation, policy), nil, nil
}

func decodeAnthropicBaseRequest(wire anthropicRequestWire) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream,
		Sampling: llmprotocol.Sampling{
			Temperature: wire.Temperature, TopP: wire.TopP, TopK: wire.TopK,
			MaxOutputTokens: llmprotocol.Int64(wire.MaxTokens), Stop: append([]string(nil), wire.StopSequences...),
		},
		Trusted: llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.AnthropicMessagesV1},
	}
	if wire.Metadata != nil && wire.Metadata.UserID != "" {
		request.EndUserID = wire.Metadata.UserID
	}
	return request
}

func decodeAnthropicThinking(thinking *anthropicThinkingWire, request *llmprotocol.Request) error {
	if thinking == nil {
		return nil
	}
	switch thinking.Type {
	case "enabled":
		request.ReasoningBudgetTokens = llmprotocol.Int64(thinking.BudgetTokens)
	case "disabled", "":
	default:
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_thinking_mode",
			"Anthropic thinking mode is unsupported", nil,
		)
	}
	return nil
}

func decodeAnthropicSystem(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	contents, err := decodeAnthropicContent(raw, policy)
	if err != nil {
		return err
	}
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: contents}}
	return nil
}

func decodeAnthropicMessages(messages []anthropicMessageWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	for index, messageWire := range messages {
		message, err := decodeAnthropicMessage(messageWire, index, policy)
		if err != nil {
			return err
		}
		request.Messages = append(request.Messages, message...)
	}
	return nil
}

func decodeAnthropicTools(tools []anthropicToolWire, request *llmprotocol.Request) error {
	for _, toolWire := range tools {
		if toolWire.Type != "" && toolWire.Type != "custom" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only custom tools enter the model protocol", nil)
		}
		if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
			"tools.allowed_callers":       toolWire.AllowedCallers,
			"tools.cache_control":         toolWire.CacheControl,
			"tools.defer_loading":         toolWire.DeferLoading,
			"tools.eager_input_streaming": toolWire.EagerInputStreaming,
			"tools.input_examples":        toolWire.InputExamples,
		}); err != nil {
			return err
		}
		schema := toolWire.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{Name: toolWire.Name, Description: toolWire.Description, InputSchema: schema, Strict: toolWire.Strict})
	}
	return nil
}

func decodeAnthropicToolChoice(choice *anthropicToolChoiceWire, request *llmprotocol.Request) error {
	if choice == nil {
		return nil
	}
	switch choice.Type {
	case "auto":
		request.ToolChoice.Mode = llmprotocol.ToolChoiceAuto
	case "none":
		request.ToolChoice.Mode = llmprotocol.ToolChoiceNone
	case "any":
		request.ToolChoice.Mode = llmprotocol.ToolChoiceRequired
	case "tool":
		request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: choice.Name}
	default:
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Anthropic tool choice is invalid", nil)
	}
	if choice.DisableParallelToolUse != nil {
		parallel := !*choice.DisableParallelToolUse
		request.ParallelToolCalls = &parallel
	}
	return nil
}

func decodeAnthropicMessage(wire anthropicMessageWire, messageIndex int, policy llmprotocol.Policy) ([]llmprotocol.Message, error) {
	role, err := canonicalRole(wire.Role)
	if err != nil || role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_anthropic_role", "Anthropic message role must be user or assistant", err)
	}
	contents, err := decodeAnthropicContent(wire.Content, policy)
	if err != nil {
		return nil, err
	}
	result := make([]llmprotocol.Message, 0, 3)
	ordinary := make([]llmprotocol.Content, 0, len(contents))
	flush := func() {
		if len(ordinary) == 0 {
			return
		}
		result = append(result, llmprotocol.Message{Role: role, Content: ordinary})
		ordinary = nil
	}
	for blockIndex, content := range contents {
		if content.Kind == llmprotocol.ContentToolResult {
			flush()
			result = append(result, llmprotocol.Message{ID: llmprotocol.StableID("anthropic-message", fmt.Sprint(messageIndex), fmt.Sprint(blockIndex)), Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{content}})
			continue
		}
		ordinary = append(ordinary, content)
	}
	flush()
	return result, nil
}

func decodeAnthropicContent(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
	}
	var blockBodies []json.RawMessage
	if err := decodeWire(raw, &blockBodies, policy); err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(blockBodies))
	for _, blockBody := range blockBodies {
		content, err := decodeAnthropicContentBlock(blockBody, policy)
		if err != nil {
			return nil, err
		}
		contents = append(contents, content)
	}
	return contents, nil
}

func decodeAnthropicContentBlock(body json.RawMessage, policy llmprotocol.Policy) (llmprotocol.Content, error) {
	typeName, err := anthropicContentType(body)
	if err != nil {
		return llmprotocol.Content{}, err
	}
	var block anthropicContentWire
	if err := decodeWire(body, &block, policy); err != nil {
		return llmprotocol.Content{}, err
	}
	if err := validateAnthropicContentExtensions(block); err != nil {
		return llmprotocol.Content{}, err
	}
	return decodeAnthropicTypedContent(typeName, block, policy)
}

func anthropicContentType(body json.RawMessage) (string, error) {
	var discriminator struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(body, &discriminator); err != nil || discriminator.Type == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "content_type_required", "Anthropic content type is required", err)
	}
	switch discriminator.Type {
	case "text", "thinking", "image", "document", "tool_use", "tool_result":
		return discriminator.Type, nil
	case "redacted_thinking":
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning", "redacted reasoning cannot be translated", nil)
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic content type is unsupported", nil)
	}
}

func validateAnthropicContentExtensions(block anthropicContentWire) error {
	if len(block.Citations) > 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_citations", "Anthropic citations are not supported by the neutral contract", nil)
	}
	if len(block.CacheControl) > 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_cache_control", "Anthropic content cache control is not supported by the neutral contract", nil)
	}
	return rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"content.caller":          block.Caller,
		"content.context":         block.Context,
		"content.title":           block.Title,
		"content.toolset_name":    block.ToolsetName,
		"content.transformations": block.Transformations,
	})
}

func decodeAnthropicTypedContent(
	typeName string,
	block anthropicContentWire,
	policy llmprotocol.Policy,
) (llmprotocol.Content, error) {
	switch typeName {
	case "text":
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: block.Text}, nil
	case "thinking":
		return llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: block.Thinking, Signature: block.Signature}, nil
	case "image", "document":
		return decodeAnthropicMediaContent(typeName, block.Source)
	case "tool_use":
		arguments := string(block.Input)
		if len(block.Input) == 0 {
			arguments = `{}`
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: block.ID, Name: block.Name, Arguments: arguments}}, nil
	case "tool_result":
		resultContent, err := decodeAnthropicContent(block.Content, policy)
		if err != nil {
			return llmprotocol.Content{}, err
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: block.ToolUseID, Content: resultContent, IsError: block.IsError}}, nil
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic content type is unsupported", nil)
	}
}

func decodeAnthropicMediaContent(typeName string, source *anthropicMediaSourceWire) (llmprotocol.Content, error) {
	if source == nil {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "media_source_required", "Anthropic media source is required", nil)
	}
	if len(source.Content) > 0 {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_document_content_source", "Anthropic document content sources are unsupported", nil)
	}
	kind := llmprotocol.ContentImage
	if typeName == "document" {
		kind = llmprotocol.ContentFile
	}
	return llmprotocol.Content{Kind: kind, MediaType: source.MediaType, Data: source.Data, URL: source.URL, FileID: source.FileID}, nil
}

func (AnthropicMessagesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	diagnostics, validationErr := anthropicRequestDiagnostics(request, policy)
	if validationErr != nil {
		return nil, diagnostics, validationErr
	}
	wire, baseErr := encodeAnthropicBaseRequest(request)
	if baseErr != nil {
		return nil, diagnostics, baseErr
	}
	if instructionErr := encodeAnthropicInstructions(&wire, request, policy, &diagnostics); instructionErr != nil {
		return nil, diagnostics, instructionErr
	}
	if messagesErr := appendAnthropicMessages(&wire, request.Messages); messagesErr != nil {
		return nil, diagnostics, messagesErr
	}
	appendAnthropicTools(&wire, request.Tools)
	if toolChoiceErr := encodeAnthropicToolChoice(&wire, request); toolChoiceErr != nil {
		return nil, diagnostics, toolChoiceErr
	}
	body, encodeErr := marshalWire(wire)
	return body, diagnostics, encodeErr
}

func anthropicRequestDiagnostics(request llmprotocol.Request, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if err := appendAnthropicContentDiagnostics(&diagnostics, request, policy); err != nil {
		return diagnostics, err
	}
	if err := appendAnthropicStateDiagnostics(&diagnostics, request, policy); err != nil {
		return diagnostics, err
	}
	return diagnostics, appendAnthropicOutputDiagnostics(&diagnostics, request, policy)
}

func appendAnthropicContentDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	request llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	contentGroups := make([][]llmprotocol.Content, 0, len(request.Instructions)+len(request.Messages))
	for _, instruction := range request.Instructions {
		contentGroups = append(contentGroups, instruction.Content)
	}
	for _, message := range request.Messages {
		contentGroups = append(contentGroups, message.Content)
	}
	for _, contents := range contentGroups {
		contentDiagnostics, err := anthropicContentDiagnostics(contents, request.Trusted.SourceFormat, policy)
		*diagnostics = appendDiagnostics(*diagnostics, contentDiagnostics, policy.Limits.Diagnostics)
		if err != nil {
			return err
		}
	}
	return nil
}

func appendAnthropicStateDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	request llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	if request.PreviousResponseID != "" || request.ConversationID != "" || request.Store != nil || request.Truncation != "" {
		return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "conversation_state", "Messages has no stateful response reference")
	}
	return nil
}

func appendAnthropicOutputDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	request llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	if request.OutputFormat.Kind == llmprotocol.OutputJSONObject || request.OutputFormat.Kind == llmprotocol.OutputJSONSchema {
		return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "output_format", "Messages has no native strict output format")
	}
	return nil
}

func encodeAnthropicBaseRequest(request llmprotocol.Request) (anthropicRequestWire, error) {
	wire := anthropicRequestWire{
		Model: request.Model, Stream: request.Stream, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, TopK: request.Sampling.TopK, StopSequences: append([]string(nil), request.Sampling.Stop...),
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens <= 0 {
		return anthropicRequestWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "max_tokens_required", "Anthropic Messages requires max output tokens", nil)
	}
	wire.MaxTokens = *request.Sampling.MaxOutputTokens
	if request.ReasoningBudgetTokens != nil {
		wire.Thinking = &anthropicThinkingWire{Type: "enabled", BudgetTokens: *request.ReasoningBudgetTokens}
	}
	if userID := request.EndUserID; userID != "" {
		wire.Metadata = &struct {
			UserID string `json:"user_id,omitempty"`
		}{UserID: userID}
	}
	return wire, nil
}

func encodeAnthropicInstructions(
	wire *anthropicRequestWire,
	request llmprotocol.Request,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) error {
	if len(request.Instructions) == 0 {
		return nil
	}
	contents := make([]llmprotocol.Content, 0)
	for _, instruction := range request.Instructions {
		if instruction.Role == llmprotocol.RoleDeveloper {
			if err := appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "instructions.role", "Messages cannot preserve developer authority"); err != nil {
				return err
			}
		}
		contents = append(contents, instruction.Content...)
	}
	encoded, err := encodeAnthropicContent(contents)
	if err == nil {
		wire.System = encoded
	}
	return err
}

func appendAnthropicMessages(wire *anthropicRequestWire, messages []llmprotocol.Message) error {
	for _, message := range messages {
		encoded, err := encodeAnthropicMessage(message)
		if err != nil {
			return err
		}
		wire.Messages = append(wire.Messages, encoded...)
	}
	return nil
}

func appendAnthropicTools(wire *anthropicRequestWire, tools []llmprotocol.Tool) {
	for _, tool := range tools {
		wire.Tools = append(wire.Tools, anthropicToolWire{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, Strict: tool.Strict})
	}
}

func encodeAnthropicToolChoice(wire *anthropicRequestWire, request llmprotocol.Request) error {
	if request.ToolChoice.Mode != "" {
		wire.ToolChoice = &anthropicToolChoiceWire{}
		switch request.ToolChoice.Mode {
		case llmprotocol.ToolChoiceAuto:
			wire.ToolChoice.Type = "auto"
		case llmprotocol.ToolChoiceNone:
			wire.ToolChoice.Type = "none"
		case llmprotocol.ToolChoiceRequired:
			wire.ToolChoice.Type = "any"
		case llmprotocol.ToolChoiceNamed:
			wire.ToolChoice.Type, wire.ToolChoice.Name = "tool", request.ToolChoice.Name
		default:
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
		}
		if request.ParallelToolCalls != nil {
			disable := !*request.ParallelToolCalls
			wire.ToolChoice.DisableParallelToolUse = &disable
		}
	}
	return nil
}

func encodeAnthropicMessage(message llmprotocol.Message) ([]anthropicMessageWire, error) {
	role := message.Role
	if role == llmprotocol.RoleSystem || role == llmprotocol.RoleDeveloper {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "system_message_position", "system and developer messages must be normalized as instructions", nil)
	}
	if role == llmprotocol.RoleTool {
		role = llmprotocol.RoleUser
	}
	if role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_role", "message role cannot be encoded as Anthropic Messages", nil)
	}
	content, err := encodeAnthropicContent(message.Content)
	if err != nil {
		return nil, err
	}
	return []anthropicMessageWire{{Role: string(role), Content: content}}, nil
}

func encodeAnthropicContent(contents []llmprotocol.Content) (json.RawMessage, error) {
	blocks := make([]anthropicContentWire, 0, len(contents))
	for _, content := range contents {
		block, err := encodeAnthropicContentBlock(content)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	return json.Marshal(blocks)
}

func encodeAnthropicContentBlock(content llmprotocol.Content) (anthropicContentWire, error) {
	switch content.Kind {
	case llmprotocol.ContentText:
		return anthropicContentWire{Type: "text", Text: content.Text}, nil
	case llmprotocol.ContentReasoning:
		return anthropicContentWire{Type: "thinking", Thinking: content.Text, Signature: content.Signature}, nil
	case llmprotocol.ContentImage, llmprotocol.ContentFile:
		return encodeAnthropicMediaBlock(content), nil
	case llmprotocol.ContentToolCall:
		return encodeAnthropicToolCallBlock(content.ToolCall)
	case llmprotocol.ContentToolResult:
		return encodeAnthropicToolResultBlock(content.ToolResult)
	case llmprotocol.ContentRefusal:
		return anthropicContentWire{Type: "text", Text: content.Text}, nil
	default:
		return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as Anthropic Messages", nil)
	}
}

func encodeAnthropicMediaBlock(content llmprotocol.Content) anthropicContentWire {
	typeName := "image"
	if content.Kind == llmprotocol.ContentFile {
		typeName = "document"
	}
	sourceType := "base64"
	if content.URL != "" {
		sourceType = "url"
	} else if content.FileID != "" {
		sourceType = "file"
	}
	return anthropicContentWire{Type: typeName, Source: &anthropicMediaSourceWire{
		Type: sourceType, MediaType: content.MediaType, Data: content.Data,
		URL: content.URL, FileID: content.FileID,
	}}
}

func encodeAnthropicToolCallBlock(call *llmprotocol.ToolCall) (anthropicContentWire, error) {
	if call == nil {
		return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call is invalid", nil)
	}
	arguments := json.RawMessage(call.Arguments)
	if !json.Valid(arguments) {
		return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_arguments", "tool arguments must be JSON", nil)
	}
	return anthropicContentWire{Type: "tool_use", ID: call.ID, Name: call.Name, Input: arguments}, nil
}

func encodeAnthropicToolResultBlock(result *llmprotocol.ToolResult) (anthropicContentWire, error) {
	if result == nil {
		return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result is invalid", nil)
	}
	content, err := encodeAnthropicContent(result.Content)
	if err != nil {
		return anthropicContentWire{}, err
	}
	return anthropicContentWire{Type: "tool_result", ToolUseID: result.CallID, Content: content, IsError: result.IsError}, nil
}

func anthropicContentDiagnostics(contents []llmprotocol.Content, source llmprotocol.WireFormat, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	for _, content := range contents {
		if len(content.Citations) > 0 {
			if err := appendLossy(&diagnostics, policy, source, llmprotocol.AnthropicMessagesV1, "content.citations", "Messages cannot represent URL citations"); err != nil {
				return diagnostics, err
			}
		}
		if content.Kind == llmprotocol.ContentRefusal {
			if err := appendLossy(&diagnostics, policy, source, llmprotocol.AnthropicMessagesV1, "content.refusal", "Messages represents refusal as ordinary text"); err != nil {
				return diagnostics, err
			}
		}
		if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
			nested, err := anthropicContentDiagnostics(content.ToolResult.Content, source, policy)
			diagnostics = appendDiagnostics(diagnostics, nested, policy.Limits.Diagnostics)
			if err != nil {
				return diagnostics, err
			}
		}
	}
	return diagnostics, nil
}

type anthropicResponseWire struct {
	ID           string              `json:"id"`
	Type         string              `json:"type,omitempty"`
	Role         string              `json:"role,omitempty"`
	Model        string              `json:"model,omitempty"`
	Content      json.RawMessage     `json:"content,omitempty"`
	StopReason   *string             `json:"stop_reason,omitempty"`
	StopSequence *string             `json:"stop_sequence,omitempty"`
	Usage        *anthropicUsageWire `json:"usage,omitempty"`
	Error        *anthropicErrorWire `json:"error,omitempty"`
	Container    json.RawMessage     `json:"container,omitempty"`
	StopDetails  json.RawMessage     `json:"stop_details,omitempty"`
}

type anthropicUsageWire struct {
	InputTokens              int64           `json:"input_tokens"`
	OutputTokens             int64           `json:"output_tokens"`
	CacheCreationInputTokens int64           `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64           `json:"cache_read_input_tokens,omitempty"`
	CacheCreation            json.RawMessage `json:"cache_creation,omitempty"`
	InferenceGeo             string          `json:"inference_geo,omitempty"`
	OutputTokensDetails      *struct {
		ThinkingTokens int64 `json:"thinking_tokens"`
	} `json:"output_tokens_details,omitempty"`
	ServerToolUse json.RawMessage `json:"server_tool_use,omitempty"`
	ServiceTier   string          `json:"service_tier,omitempty"`
}

type anthropicErrorWire struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicTransportErrorWire struct {
	Type      string              `json:"type"`
	Error     *anthropicErrorWire `json:"error"`
	RequestID string              `json:"request_id,omitempty"`
}

func (AnthropicMessagesCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire anthropicResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	var diagnostics llmprotocol.Diagnostics
	if len(wire.Container) > 0 {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "container", "container execution metadata is not model output")
	}
	if len(wire.StopDetails) > 0 {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "stop_details", "structured refusal detail has no neutral representation")
	}
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Type), Code: wire.Error.Type, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	contents, err := decodeAnthropicContent(wire.Content, policy)
	if err != nil && wire.Error == nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if len(contents) > 0 {
		response.Output = []llmprotocol.OutputItem{{ID: llmprotocol.StableID("anthropic-response", wire.ID), Role: llmprotocol.RoleAssistant, Content: contents}}
	}
	if wire.StopReason != nil && response.Error == nil {
		response.SourceStopReason = *wire.StopReason
		response.StopReason = decodeAnthropicStop(*wire.StopReason)
	}
	if wire.Usage != nil {
		response.Usage = decodeAnthropicUsage(*wire.Usage)
		appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, map[string]bool{
			"usage.cache_creation":  len(wire.Usage.CacheCreation) > 0,
			"usage.inference_geo":   wire.Usage.InferenceGeo != "",
			"usage.server_tool_use": len(wire.Usage.ServerToolUse) > 0,
			"usage.service_tier":    wire.Usage.ServiceTier != "",
		}, "provider usage metadata has no neutral accounting bucket")
	}
	return response, responseEnvelope(llmprotocol.AnthropicMessagesV1, body, response.Generation, response.SourceStopReason, policy), diagnostics, nil
}

func decodeAnthropicUsage(wire anthropicUsageWire) llmprotocol.Usage {
	uncached := wire.InputTokens - wire.CacheReadInputTokens - wire.CacheCreationInputTokens
	if uncached < 0 {
		uncached = 0
	}
	reasoning := int64(0)
	if wire.OutputTokensDetails != nil {
		reasoning = wire.OutputTokensDetails.ThinkingTokens
	}
	other := wire.OutputTokens - reasoning
	if other < 0 {
		other = 0
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(uncached), InputCacheRead: authoritative(wire.CacheReadInputTokens), InputCacheWrite: authoritative(wire.CacheCreationInputTokens),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(wire.InputTokens), OutputTotal: authoritative(wire.OutputTokens), Total: llmprotocol.TokenCount{Value: llmprotocol.Int64(wire.InputTokens + wire.OutputTokens), Provenance: llmprotocol.UsageDerived},
	}
}

func (AnthropicMessagesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	if response.Error != nil {
		return encodeAnthropicError(response.Error, response.ProviderRequestID), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if response.Usage.OutputReasoning.Value != nil {
		appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1, "usage.output_reasoning", "Messages has no reasoning-token usage field")
	}
	if len(response.Alternatives) > 0 {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1, "response.alternatives", "Messages has one output sequence"); err != nil {
			return nil, diagnostics, err
		}
	}
	contents := make([]llmprotocol.Content, 0)
	for _, item := range response.Output {
		contents = append(contents, item.Content...)
	}
	contentDiagnostics, err := anthropicContentDiagnostics(contents, envelope.Format, policy)
	diagnostics = appendDiagnostics(diagnostics, contentDiagnostics, policy.Limits.Diagnostics)
	if err != nil {
		return nil, diagnostics, err
	}
	content, err := encodeAnthropicContent(contents)
	if err != nil {
		return nil, diagnostics, err
	}
	stop := encodeAnthropicStop(response.StopReason)
	wire := anthropicResponseWire{ID: response.ID, Type: "message", Role: "assistant", Model: response.Model, Content: content, StopReason: &stop, Usage: encodeAnthropicUsage(response.Usage)}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeAnthropicUsage(usage llmprotocol.Usage) *anthropicUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil {
		return nil
	}
	return &anthropicUsageWire{InputTokens: tokenValue(usage.InputTotal), OutputTokens: tokenValue(usage.OutputTotal), CacheCreationInputTokens: tokenValue(usage.InputCacheWrite), CacheReadInputTokens: tokenValue(usage.InputCacheRead)}
}

func decodeAnthropicStop(reason string) llmprotocol.StopReason {
	switch reason {
	case "end_turn":
		return llmprotocol.StopEndTurn
	case "max_tokens":
		return llmprotocol.StopMaxTokens
	case "stop_sequence":
		return llmprotocol.StopSequence
	case "tool_use":
		return llmprotocol.StopToolCall
	case "refusal":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeAnthropicStop(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "max_tokens"
	case llmprotocol.StopSequence:
		return "stop_sequence"
	case llmprotocol.StopToolCall:
		return "tool_use"
	case llmprotocol.StopContentFilter:
		return "refusal"
	default:
		return "end_turn"
	}
}

func encodeAnthropicError(protocolError *llmprotocol.ProtocolError, requestID string) []byte {
	return AnthropicMessagesCodec{}.EncodeTransportError(llmprotocol.TransportError{
		Error: protocolError, ProviderRequestID: requestID,
	})
}

func (AnthropicMessagesCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire anthropicTransportErrorWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	if wire.Error == nil {
		return llmprotocol.TransportError{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_error_required",
			"upstream transport error body is missing error details",
			nil,
		)
	}
	return llmprotocol.TransportError{
		Error: &llmprotocol.ProtocolError{
			Category: decodeProviderErrorCategory(wire.Error.Type),
			Code:     wire.Error.Type, Message: wire.Error.Message,
		},
		ProviderRequestID: wire.RequestID,
	}, nil, nil
}

func (AnthropicMessagesCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	protocolError := transportError.Error
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	wire := anthropicTransportErrorWire{
		Type: "error", RequestID: transportError.ProviderRequestID,
		Error: &anthropicErrorWire{Type: canonicalAnthropicErrorType(protocolError), Message: protocolError.Message},
	}
	body, _ := json.Marshal(wire)
	return body
}

func canonicalAnthropicErrorType(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return "api_error"
	}
	if canonicalAnthropicErrorTypeMatchesCategory(protocolError.Code, protocolError.Category) {
		return protocolError.Code
	}
	switch protocolError.Category {
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_request_error"
	case llmprotocol.ErrorAuthentication:
		return "authentication_error"
	case llmprotocol.ErrorPermission:
		return "permission_error"
	case llmprotocol.ErrorNotFound:
		return "not_found_error"
	case llmprotocol.ErrorConflict:
		return "conflict_error"
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_error"
	case llmprotocol.ErrorUpstreamTimeout:
		return "timeout_error"
	default:
		return "api_error"
	}
}

func canonicalAnthropicErrorTypeMatchesCategory(code string, category llmprotocol.ErrorCategory) bool {
	categories := map[string][]llmprotocol.ErrorCategory{
		"invalid_request_error": {llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature},
		"request_too_large":     {llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature},
		"authentication_error":  {llmprotocol.ErrorAuthentication},
		"billing_error":         {llmprotocol.ErrorUpstreamUnavailable},
		"permission_error":      {llmprotocol.ErrorPermission},
		"not_found_error":       {llmprotocol.ErrorNotFound},
		"conflict_error":        {llmprotocol.ErrorConflict},
		"rate_limit_error":      {llmprotocol.ErrorRateLimited},
		"timeout_error":         {llmprotocol.ErrorUpstreamTimeout},
		"api_error":             {llmprotocol.ErrorUpstreamUnavailable, llmprotocol.ErrorInternal},
		"overloaded_error":      {llmprotocol.ErrorUpstreamUnavailable, llmprotocol.ErrorInternal},
	}
	for _, allowed := range categories[code] {
		if allowed == category {
			return true
		}
	}
	return false
}
