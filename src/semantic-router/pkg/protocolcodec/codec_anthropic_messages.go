package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const defaultAnthropicMaxOutputTokens int64 = 4096

type AnthropicMessagesCodec struct{}

func (AnthropicMessagesCodec) Format() llmprotocol.WireFormat { return llmprotocol.AnthropicMessagesV1 }
func (AnthropicMessagesCodec) Stateless() bool                { return true }
func (AnthropicMessagesCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStrictToolSchema,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
		llmprotocol.CapabilityCacheDirectives, llmprotocol.CapabilityReasoningDisable,
		llmprotocol.CapabilityReasoningBudget, llmprotocol.CapabilityReasoningEffort,
		llmprotocol.CapabilityStrictJSONSchema, llmprotocol.CapabilitySamplingTopK,
		llmprotocol.CapabilityStopSequences, llmprotocol.CapabilityReasoningAdaptive,
		llmprotocol.CapabilityReasoningSignature, llmprotocol.CapabilityReasoningDisplay,
		llmprotocol.CapabilityMatchedStopSequence,
	)
}

type anthropicRequestWire struct {
	Model         string                     `json:"model"`
	System        json.RawMessage            `json:"system,omitempty"`
	Messages      []anthropicMessageWire     `json:"messages"`
	MaxTokens     *int64                     `json:"max_tokens"`
	Temperature   *float64                   `json:"temperature,omitempty"`
	TopP          *float64                   `json:"top_p,omitempty"`
	TopK          *int64                     `json:"top_k,omitempty"`
	StopSequences []string                   `json:"stop_sequences,omitempty"`
	Tools         json.RawMessage            `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoiceWire   `json:"tool_choice,omitempty"`
	Metadata      *anthropicMetadataWire     `json:"metadata,omitempty"`
	Thinking      *anthropicThinkingWire     `json:"thinking,omitempty"`
	Stream        bool                       `json:"stream,omitempty"`
	InferenceGeo  json.RawMessage            `json:"inference_geo,omitempty"`
	Container     json.RawMessage            `json:"container,omitempty"`
	CacheControl  json.RawMessage            `json:"cache_control,omitempty"`
	OutputConfig  *anthropicOutputConfigWire `json:"output_config,omitempty"`
	ServiceTier   json.RawMessage            `json:"service_tier,omitempty"`
}

type anthropicMessageWire struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicContentWire struct {
	Type            string                     `json:"type"`
	Text            string                     `json:"text,omitempty"`
	Thinking        string                     `json:"thinking,omitempty"`
	Signature       string                     `json:"signature,omitempty"`
	Data            string                     `json:"data,omitempty"`
	ID              string                     `json:"id,omitempty"`
	Name            string                     `json:"name,omitempty"`
	Input           json.RawMessage            `json:"input,omitempty"`
	ToolUseID       string                     `json:"tool_use_id,omitempty"`
	Content         json.RawMessage            `json:"content,omitempty"`
	IsError         *bool                      `json:"is_error,omitempty"`
	Source          *anthropicMediaSourceWire  `json:"source,omitempty"`
	Citations       json.RawMessage            `json:"citations,omitempty"`
	CacheControl    *anthropicCacheControlWire `json:"cache_control,omitempty"`
	Caller          json.RawMessage            `json:"caller,omitempty"`
	Context         json.RawMessage            `json:"context,omitempty"`
	Title           json.RawMessage            `json:"title,omitempty"`
	ToolsetName     json.RawMessage            `json:"toolset_name,omitempty"`
	Transformations json.RawMessage            `json:"transformations,omitempty"`
	FileID          string                     `json:"file_id,omitempty"`
}

func (wire anthropicContentWire) MarshalJSON() ([]byte, error) {
	type wireAlias anthropicContentWire
	body, err := json.Marshal(wireAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "text":
		object["text"], _ = json.Marshal(wire.Text)
	case "thinking":
		object["thinking"], _ = json.Marshal(wire.Thinking)
	}
	return json.Marshal(object)
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
	Name                string                     `json:"name"`
	Description         string                     `json:"description,omitempty"`
	InputSchema         json.RawMessage            `json:"input_schema"`
	Strict              *bool                      `json:"strict,omitempty"`
	Type                string                     `json:"type,omitempty"`
	AllowedCallers      json.RawMessage            `json:"allowed_callers,omitempty"`
	CacheControl        *anthropicCacheControlWire `json:"cache_control,omitempty"`
	DeferLoading        json.RawMessage            `json:"defer_loading,omitempty"`
	EagerInputStreaming json.RawMessage            `json:"eager_input_streaming,omitempty"`
	InputExamples       json.RawMessage            `json:"input_examples,omitempty"`
}

type anthropicCacheControlWire struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type anthropicMetadataWire struct {
	UserID string `json:"user_id,omitempty"`
}

type anthropicToolChoiceWire struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type anthropicThinkingWire struct {
	Type         string `json:"type"`
	BudgetTokens *int64 `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

type anthropicOutputConfigWire struct {
	Effort string                         `json:"effort,omitempty"`
	Format *anthropicJSONOutputFormatWire `json:"format,omitempty"`
}

type anthropicJSONOutputFormatWire struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

func (AnthropicMessagesCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire anthropicRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateAnthropicRequestWire(wire); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request := decodeAnthropicBaseRequest(wire)
	if err := decodeAnthropicRequestFields(wire, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.AnthropicMessagesV1, body, request.Generation, policy), nil, nil
}

func validateAnthropicRequestWire(wire anthropicRequestWire) error {
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"inference_geo": wire.InferenceGeo, "container": wire.Container,
		"cache_control": wire.CacheControl,
		"service_tier":  wire.ServiceTier,
	}); err != nil {
		return err
	}
	if wire.MaxTokens == nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"max_tokens_required",
			"Anthropic max_tokens is required",
			nil,
		)
	}
	if wire.Temperature != nil && *wire.Temperature > 1 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_anthropic_temperature",
			"Anthropic temperature must be between 0 and 1",
			nil,
		)
	}
	if len(wire.Messages) == 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"messages_required",
			"Anthropic messages must contain at least one item",
			nil,
		)
	}
	return nil
}

func decodeAnthropicRequestFields(wire anthropicRequestWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if err := decodeAnthropicOutputConfig(wire.OutputConfig, request); err != nil {
		return err
	}
	if err := decodeAnthropicThinking(wire.Thinking, request); err != nil {
		return err
	}
	if err := validateAnthropicReasoningBudget(*request, *wire.MaxTokens, llmprotocol.ErrorInvalidRequest); err != nil {
		return err
	}
	if err := decodeAnthropicSystem(wire.System, request, policy); err != nil {
		return err
	}
	if err := decodeAnthropicMessages(wire.Messages, request, policy); err != nil {
		return err
	}
	if err := decodeAnthropicTools(wire.Tools, request, policy); err != nil {
		return err
	}
	return decodeAnthropicToolChoice(wire.ToolChoice, request)
}

func decodeAnthropicOutputConfig(output *anthropicOutputConfigWire, request *llmprotocol.Request) error {
	if output == nil {
		return nil
	}
	if output.Effort != "" {
		switch output.Effort {
		case "low", "medium", "high", "xhigh", "max":
			request.ReasoningEffort = output.Effort
		default:
			return llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"invalid_reasoning_effort",
				"Anthropic output effort is invalid",
				nil,
			)
		}
	}
	if output.Format == nil {
		return nil
	}
	if output.Format.Type != "json_schema" || len(output.Format.Schema) == 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_output_format",
			"Anthropic output format must contain a JSON Schema",
			nil,
		)
	}
	request.OutputFormat = llmprotocol.OutputFormat{
		Kind: llmprotocol.OutputJSONSchema,
		// Anthropic does not name output schemas. The stable neutral name is used
		// only when translating to an OpenAI wire contract that requires one.
		Name:   "structured_output",
		Strict: llmprotocol.Bool(true),
		Schema: append(json.RawMessage(nil), output.Format.Schema...),
	}
	return nil
}

func decodeAnthropicBaseRequest(wire anthropicRequestWire) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream,
		Sampling: llmprotocol.Sampling{
			Temperature: wire.Temperature, TopP: wire.TopP, TopK: wire.TopK,
			MaxOutputTokens: llmprotocol.Int64(*wire.MaxTokens), Stop: append([]string(nil), wire.StopSequences...),
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
		if thinking.BudgetTokens == nil {
			return llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"reasoning_budget_required",
				"enabled Anthropic thinking requires budget_tokens",
				nil,
			)
		}
		request.ReasoningMode = llmprotocol.ReasoningModeEnabled
		request.ReasoningBudgetTokens = llmprotocol.Int64(*thinking.BudgetTokens)
		request.ReasoningDisplay = thinking.Display
	case "disabled":
		if thinking.BudgetTokens != nil || thinking.Display != "" {
			return llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"invalid_disabled_thinking",
				"disabled Anthropic thinking cannot include budget_tokens or display",
				nil,
			)
		}
		request.ReasoningMode = llmprotocol.ReasoningModeDisabled
	case "adaptive":
		if thinking.BudgetTokens != nil {
			return llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"invalid_adaptive_thinking",
				"adaptive Anthropic thinking cannot include budget_tokens",
				nil,
			)
		}
		request.ReasoningMode = llmprotocol.ReasoningModeAdaptive
		request.ReasoningDisplay = thinking.Display
	case "":
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"thinking_type_required",
			"Anthropic thinking type is required",
			nil,
		)
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
	contents, err := decodeAnthropicRequestContent(raw, policy)
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

func decodeAnthropicTools(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
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
		if err := json.Unmarshal(toolBody, &discriminator); err != nil {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Anthropic tool is invalid", err)
		}
		if discriminator.Type != "" && discriminator.Type != "custom" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only custom tools enter the model protocol", nil)
		}
		var toolWire anthropicToolWire
		if err := decodeWireValue(toolBody, &toolWire, policy); err != nil {
			return err
		}
		if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
			"tools.allowed_callers":       toolWire.AllowedCallers,
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
		request.Tools[len(request.Tools)-1].Cache = decodeAnthropicCacheControl(toolWire.CacheControl)
	}
	return nil
}

func decodeAnthropicToolChoice(choice *anthropicToolChoiceWire, request *llmprotocol.Request) error {
	if choice == nil {
		return nil
	}
	toolChoice, err := decodeAnthropicToolChoiceVariant(*choice)
	if err != nil {
		return err
	}
	request.ToolChoice = toolChoice
	if choice.DisableParallelToolUse != nil {
		parallel := !*choice.DisableParallelToolUse
		request.ParallelToolCalls = &parallel
	}
	return nil
}

func decodeAnthropicToolChoiceVariant(choice anthropicToolChoiceWire) (llmprotocol.ToolChoice, error) {
	switch choice.Type {
	case "auto":
		if choice.Name != "" {
			return llmprotocol.ToolChoice{}, invalidAnthropicToolChoiceVariant("name")
		}
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}, nil
	case "none":
		if choice.Name != "" {
			return llmprotocol.ToolChoice{}, invalidAnthropicToolChoiceVariant("name")
		}
		if choice.DisableParallelToolUse != nil {
			return llmprotocol.ToolChoice{}, invalidAnthropicToolChoiceVariant("disable_parallel_tool_use")
		}
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}, nil
	case "any":
		if choice.Name != "" {
			return llmprotocol.ToolChoice{}, invalidAnthropicToolChoiceVariant("name")
		}
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceRequired}, nil
	case "tool":
		if choice.Name == "" {
			return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Anthropic named tool choice requires a name", nil)
		}
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: choice.Name}, nil
	default:
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Anthropic tool choice is invalid", nil)
	}
}

func invalidAnthropicToolChoiceVariant(field string) error {
	return llmprotocol.NewError(
		llmprotocol.ErrorInvalidRequest,
		"invalid_tool_choice_variant",
		"Anthropic tool choice includes a field from a different union variant: "+field,
		nil,
	)
}

func decodeAnthropicMessage(wire anthropicMessageWire, messageIndex int, policy llmprotocol.Policy) ([]llmprotocol.Message, error) {
	role, err := canonicalRole(wire.Role)
	if err != nil || role != llmprotocol.RoleSystem && role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_anthropic_role", "Anthropic message role must be system, user, or assistant", err)
	}
	contents, err := decodeAnthropicRequestContent(wire.Content, policy)
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

func decodeAnthropicRequestContent(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
	}
	return decodeAnthropicContentArray(raw, policy, false, decodeAnthropicRequestContentBlock)
}

func decodeAnthropicResponseContent(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_response_content",
			"Anthropic response content must be an array",
			nil,
		)
	}
	return decodeAnthropicContentArray(raw, policy, true, decodeAnthropicResponseContentBlock)
}

type anthropicContentBlockDecoder func(json.RawMessage, llmprotocol.Policy) (llmprotocol.Content, error)

func decodeAnthropicContentArray(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	providerOutput bool,
	decodeBlock anthropicContentBlockDecoder,
) ([]llmprotocol.Content, error) {
	var blockBodies []json.RawMessage
	var err error
	if providerOutput {
		err = decodeProviderValue(raw, &blockBodies, policy)
	} else {
		err = decodeWireValue(raw, &blockBodies, policy)
	}
	if err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(blockBodies))
	for _, blockBody := range blockBodies {
		content, err := decodeBlock(blockBody, policy)
		if err != nil {
			return nil, err
		}
		contents = append(contents, content)
	}
	return contents, nil
}

func decodeAnthropicRequestContentBlock(body json.RawMessage, policy llmprotocol.Policy) (llmprotocol.Content, error) {
	typeName, err := anthropicRequestContentType(body)
	if err != nil {
		return llmprotocol.Content{}, err
	}
	return decodeAnthropicContentBlock(body, typeName, policy, false)
}

func decodeAnthropicResponseContentBlock(body json.RawMessage, policy llmprotocol.Policy) (llmprotocol.Content, error) {
	typeName, err := anthropicResponseContentType(body)
	if err != nil {
		return llmprotocol.Content{}, err
	}
	return decodeAnthropicContentBlock(body, typeName, policy, true)
}

func decodeAnthropicContentBlock(
	body json.RawMessage,
	typeName string,
	policy llmprotocol.Policy,
	providerOutput bool,
) (llmprotocol.Content, error) {
	var block anthropicContentWire
	var err error
	if providerOutput {
		err = decodeProviderValue(body, &block, policy)
	} else {
		err = decodeWireValue(body, &block, policy)
	}
	if err != nil {
		return llmprotocol.Content{}, err
	}
	if err := validateAnthropicContentVariant(body, typeName, providerOutput); err != nil {
		return llmprotocol.Content{}, err
	}
	if err := validateAnthropicContentExtensions(block); err != nil {
		return llmprotocol.Content{}, err
	}
	return decodeAnthropicTypedContent(typeName, block, policy)
}

func validateAnthropicContentVariant(body json.RawMessage, typeName string, providerOutput bool) error {
	allowedByType := map[string][]string{
		"text":        {"cache_control", "citations", "text", "type"},
		"thinking":    {"signature", "thinking", "type"},
		"image":       {"cache_control", "source", "transformations", "type"},
		"document":    {"cache_control", "citations", "context", "source", "title", "type"},
		"tool_use":    {"cache_control", "caller", "id", "input", "name", "toolset_name", "type"},
		"tool_result": {"cache_control", "content", "is_error", "tool_use_id", "toolset_name", "type"},
	}
	if providerOutput {
		allowedByType = map[string][]string{
			"text":     {"citations", "text", "type"},
			"thinking": {"signature", "thinking", "type"},
			"tool_use": {"caller", "id", "input", "name", "toolset_name", "type"},
		}
	}
	allowed, recognized := allowedByType[typeName]
	if !recognized {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return err
	}
	if err := requireAnthropicContentFields(object, typeName, providerOutput); err != nil {
		return err
	}
	return rejectAnthropicContentVariantFields(object, allowed, providerOutput)
}

func requireAnthropicContentFields(object map[string]json.RawMessage, typeName string, providerOutput bool) error {
	requiredByType := map[string][]string{
		"text":        {"text"},
		"thinking":    {"thinking"},
		"image":       {"source"},
		"document":    {"source"},
		"tool_use":    {"id", "input", "name"},
		"tool_result": {"content", "tool_use_id"},
	}
	for _, name := range requiredByType[typeName] {
		if _, present := object[name]; present {
			continue
		}
		category := llmprotocol.ErrorInvalidRequest
		code := "invalid_content_variant"
		message := "Anthropic content is missing the required field: " + name
		if providerOutput {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Anthropic provider output is missing the required field: " + name
		}
		return llmprotocol.NewError(category, code, message, nil)
	}
	return nil
}

func rejectAnthropicContentVariantFields(
	object map[string]json.RawMessage,
	allowed []string,
	providerOutput bool,
) error {
	known := []string{
		"cache_control", "caller", "citations", "content", "context", "data", "file_id", "id", "input",
		"is_error", "name", "signature", "source", "text", "thinking", "title", "tool_use_id",
		"toolset_name", "transformations", "type",
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
		message := "Anthropic content includes a field from a different union variant"
		if providerOutput {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Anthropic provider output mixes content union variants"
		}
		return llmprotocol.NewError(category, code, message+": "+name, nil)
	}
	return nil
}

func anthropicContentDiscriminator(body json.RawMessage) (string, error) {
	var discriminator struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(body, &discriminator); err != nil || discriminator.Type == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "content_type_required", "Anthropic content type is required", err)
	}
	return discriminator.Type, nil
}

func anthropicRequestContentType(body json.RawMessage) (string, error) {
	typeName, err := anthropicContentDiscriminator(body)
	if err != nil {
		return "", err
	}
	switch typeName {
	case "text", "thinking", "image", "document", "tool_use", "tool_result":
		return typeName, nil
	case "redacted_thinking":
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning", "redacted reasoning cannot be translated", nil)
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic content type is unsupported", nil)
	}
}

func anthropicResponseContentType(body json.RawMessage) (string, error) {
	typeName, err := anthropicContentDiscriminator(body)
	if err != nil {
		return "", err
	}
	switch typeName {
	case "text", "thinking", "tool_use":
		return typeName, nil
	case "redacted_thinking":
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning", "redacted reasoning cannot be translated", nil)
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic response content type is unsupported", nil)
	}
}

func validateAnthropicContentExtensions(block anthropicContentWire) error {
	if len(block.Citations) > 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_citations", "Anthropic citations are not supported by the neutral contract", nil)
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
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: block.Text, Cache: decodeAnthropicCacheControl(block.CacheControl)}, nil
	case "thinking":
		if block.CacheControl != nil {
			return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_cache_directive", "Anthropic Messages cannot attach cache_control to thinking content", nil)
		}
		return llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: block.Thinking, Signature: block.Signature,
			Reasoning: llmprotocol.ReasoningScopeText,
		}, nil
	case "image", "document":
		content, err := decodeAnthropicMediaContent(typeName, block.Source)
		content.Cache = decodeAnthropicCacheControl(block.CacheControl)
		return content, err
	case "tool_use":
		arguments := string(block.Input)
		if len(block.Input) == 0 {
			arguments = `{}`
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: block.ID, Name: block.Name, Arguments: arguments}, Cache: decodeAnthropicCacheControl(block.CacheControl)}, nil
	case "tool_result":
		resultContent, err := decodeAnthropicRequestContent(block.Content, policy)
		if err != nil {
			return llmprotocol.Content{}, err
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: block.ToolUseID, Content: resultContent, IsError: block.IsError}, Cache: decodeAnthropicCacheControl(block.CacheControl)}, nil
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic content type is unsupported", nil)
	}
}

func decodeAnthropicCacheControl(cache *anthropicCacheControlWire) *llmprotocol.CacheDirective {
	if cache == nil {
		return nil
	}
	return &llmprotocol.CacheDirective{Type: cache.Type, TTL: cache.TTL}
}

func encodeAnthropicCacheControl(cache *llmprotocol.CacheDirective) *anthropicCacheControlWire {
	if cache == nil {
		return nil
	}
	return &anthropicCacheControlWire{Type: cache.Type, TTL: cache.TTL}
}

func decodeAnthropicMediaContent(typeName string, source *anthropicMediaSourceWire) (llmprotocol.Content, error) {
	if source == nil {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "media_source_required", "Anthropic media source is required", nil)
	}
	if err := validateAnthropicMediaSource(typeName, source); err != nil {
		return llmprotocol.Content{}, err
	}
	kind := llmprotocol.ContentImage
	if typeName == "document" {
		kind = llmprotocol.ContentFile
	}
	return llmprotocol.Content{Kind: kind, MediaType: source.MediaType, Data: source.Data, URL: source.URL, FileID: source.FileID}, nil
}

func validateAnthropicMediaSource(typeName string, source *anthropicMediaSourceWire) error {
	switch source.Type {
	case "base64":
		return validateAnthropicBase64Source(source)
	case "url":
		return validateAnthropicURLSource(source)
	case "file":
		return validateAnthropicFileSource(source)
	case "text", "content":
		if typeName != "document" {
			return invalidAnthropicMediaSource("Anthropic image sources do not support this source type")
		}
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_document_source",
			"Anthropic document source cannot be represented by the neutral protocol",
			nil,
		)
	case "":
		return invalidAnthropicMediaSource("Anthropic media source type is required")
	default:
		return invalidAnthropicMediaSource("Anthropic media source type is invalid")
	}
}

func validateAnthropicBase64Source(source *anthropicMediaSourceWire) error {
	if source.Data == "" || source.MediaType == "" {
		return invalidAnthropicMediaSource("Anthropic base64 media sources require data and media_type")
	}
	if source.URL != "" || source.FileID != "" || len(source.Content) > 0 {
		return invalidAnthropicMediaSource("Anthropic base64 media sources cannot contain another source variant")
	}
	return nil
}

func validateAnthropicURLSource(source *anthropicMediaSourceWire) error {
	if source.URL == "" {
		return invalidAnthropicMediaSource("Anthropic URL media sources require url")
	}
	if source.Data != "" || source.MediaType != "" || source.FileID != "" || len(source.Content) > 0 {
		return invalidAnthropicMediaSource("Anthropic URL media sources cannot contain another source variant")
	}
	return nil
}

func validateAnthropicFileSource(source *anthropicMediaSourceWire) error {
	if source.FileID == "" {
		return invalidAnthropicMediaSource("Anthropic file media sources require file_id")
	}
	if source.Data != "" || source.MediaType != "" || source.URL != "" || len(source.Content) > 0 {
		return invalidAnthropicMediaSource("Anthropic file media sources cannot contain another source variant")
	}
	return nil
}

func invalidAnthropicMediaSource(message string) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_media_source", message, nil)
}

func (AnthropicMessagesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	if err := validateAnthropicEncodableRequest(request); err != nil {
		return nil, nil, err
	}
	diagnostics, validationErr := anthropicRequestDiagnostics(request, policy)
	if validationErr != nil {
		return nil, diagnostics, validationErr
	}
	wire, diagnostics, err := buildAnthropicRequestWire(request, policy, diagnostics)
	if err != nil {
		return nil, diagnostics, err
	}
	body, encodeErr := marshalWire(wire)
	return body, diagnostics, encodeErr
}

func validateAnthropicEncodableRequest(request llmprotocol.Request) error {
	if request.Sampling.Temperature != nil && *request.Sampling.Temperature > 1 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_anthropic_temperature",
			"Anthropic Messages cannot represent a temperature above 1",
			nil,
		)
	}
	if len(request.Messages) == 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"anthropic_messages_required",
			"Anthropic Messages requires at least one conversation message",
			nil,
		)
	}
	maxTokens := int64(defaultAnthropicMaxOutputTokens)
	if request.Sampling.MaxOutputTokens != nil {
		maxTokens = *request.Sampling.MaxOutputTokens
	}
	return validateAnthropicReasoningBudget(request, maxTokens, llmprotocol.ErrorUnsupportedFeature)
}

func buildAnthropicRequestWire(
	request llmprotocol.Request,
	policy llmprotocol.Policy,
	diagnostics llmprotocol.Diagnostics,
) (anthropicRequestWire, llmprotocol.Diagnostics, error) {
	wire, baseDiagnostics, baseErr := encodeAnthropicBaseRequest(request)
	diagnostics = appendDiagnostics(diagnostics, baseDiagnostics, policy.Limits.Diagnostics)
	if baseErr != nil {
		return anthropicRequestWire{}, diagnostics, baseErr
	}
	if instructionErr := encodeAnthropicInstructions(&wire, request, policy, &diagnostics); instructionErr != nil {
		return anthropicRequestWire{}, diagnostics, instructionErr
	}
	if messagesErr := appendAnthropicMessages(&wire, request.Messages); messagesErr != nil {
		return anthropicRequestWire{}, diagnostics, messagesErr
	}
	if toolsErr := appendAnthropicTools(&wire, request.Tools); toolsErr != nil {
		return anthropicRequestWire{}, diagnostics, toolsErr
	}
	if toolChoiceErr := encodeAnthropicToolChoice(&wire, request); toolChoiceErr != nil {
		return anthropicRequestWire{}, diagnostics, toolChoiceErr
	}
	return wire, diagnostics, nil
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
	if request.ReasoningEffort == "none" || request.ReasoningEffort == "minimal" {
		return appendLossy(
			diagnostics,
			policy,
			request.Trusted.SourceFormat,
			llmprotocol.AnthropicMessagesV1,
			"reasoning_effort",
			"Messages does not define the requested reasoning effort",
		)
	}
	if request.OutputFormat.Kind == llmprotocol.OutputJSONObject {
		return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "output_format", "Messages requires an explicit JSON Schema")
	}
	if request.OutputFormat.Kind != llmprotocol.OutputJSONSchema {
		return nil
	}
	if request.OutputFormat.Strict != nil && !*request.OutputFormat.Strict {
		return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "output_format.strict", "Messages always enforces its output schema")
	}
	if request.OutputFormat.Name != "" && request.OutputFormat.Name != "structured_output" {
		appendProviderFieldOmission(diagnostics, policy, request.Trusted.SourceFormat, "output_format.name", "Messages output schemas are unnamed")
	}
	if request.OutputFormat.Description != "" {
		appendProviderFieldOmission(diagnostics, policy, request.Trusted.SourceFormat, "output_format.description", "Messages output schemas do not carry descriptions")
	}
	return nil
}

func encodeAnthropicBaseRequest(request llmprotocol.Request) (anthropicRequestWire, llmprotocol.Diagnostics, error) {
	wire := anthropicRequestWire{
		Model: request.Model, Stream: request.Stream, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, TopK: request.Sampling.TopK, StopSequences: append([]string(nil), request.Sampling.Stop...),
	}
	var diagnostics llmprotocol.Diagnostics
	if request.Sampling.MaxOutputTokens == nil {
		wire.MaxTokens = llmprotocol.Int64(defaultAnthropicMaxOutputTokens)
		diagnostics = append(diagnostics, llmprotocol.Diagnostic{
			Source: request.Trusted.SourceFormat,
			Target: llmprotocol.AnthropicMessagesV1,
			Field:  "max_tokens",
			Action: llmprotocol.DiagnosticGenerated,
			Reason: "Anthropic Messages requires an explicit output limit; the source request omitted one",
		})
	} else {
		wire.MaxTokens = llmprotocol.Int64(*request.Sampling.MaxOutputTokens)
	}
	switch request.ReasoningMode {
	case llmprotocol.ReasoningModeDisabled:
		wire.Thinking = &anthropicThinkingWire{Type: "disabled"}
	case llmprotocol.ReasoningModeAdaptive:
		wire.Thinking = &anthropicThinkingWire{Type: "adaptive", Display: request.ReasoningDisplay}
	case llmprotocol.ReasoningModeEnabled:
		if request.ReasoningBudgetTokens == nil {
			return wire, diagnostics, llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"reasoning_budget_required",
				"enabled reasoning requires a token budget",
				nil,
			)
		}
		wire.Thinking = &anthropicThinkingWire{Type: "enabled", BudgetTokens: llmprotocol.Int64(*request.ReasoningBudgetTokens), Display: request.ReasoningDisplay}
	default:
		if request.ReasoningBudgetTokens != nil {
			wire.Thinking = &anthropicThinkingWire{Type: "enabled", BudgetTokens: llmprotocol.Int64(*request.ReasoningBudgetTokens)}
		}
	}
	if userID := request.EndUserID; userID != "" {
		wire.Metadata = &anthropicMetadataWire{UserID: userID}
	}
	if request.ReasoningEffort != "" || request.OutputFormat.Kind == llmprotocol.OutputJSONSchema {
		wire.OutputConfig = &anthropicOutputConfigWire{Effort: request.ReasoningEffort}
		if request.OutputFormat.Kind == llmprotocol.OutputJSONSchema {
			wire.OutputConfig.Format = &anthropicJSONOutputFormatWire{
				Type:   "json_schema",
				Schema: append(json.RawMessage(nil), request.OutputFormat.Schema...),
			}
		}
	}
	return wire, diagnostics, nil
}

func validateAnthropicReasoningBudget(
	request llmprotocol.Request,
	maxTokens int64,
	category llmprotocol.ErrorCategory,
) error {
	if request.ReasoningBudgetTokens == nil ||
		request.ReasoningMode != "" && request.ReasoningMode != llmprotocol.ReasoningModeEnabled {
		return nil
	}
	budget := *request.ReasoningBudgetTokens
	if budget >= 1024 && budget < maxTokens {
		return nil
	}
	code := "invalid_anthropic_reasoning_budget"
	if category == llmprotocol.ErrorUnsupportedFeature {
		code = "unsupported_anthropic_reasoning_budget"
	}
	return llmprotocol.NewError(
		category,
		code,
		"Anthropic reasoning budget must be at least 1024 and less than max_tokens",
		nil,
	)
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

func appendAnthropicTools(wire *anthropicRequestWire, tools []llmprotocol.Tool) error {
	encoded := make([]anthropicToolWire, 0, len(tools))
	for _, tool := range tools {
		encoded = append(encoded, anthropicToolWire{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, Strict: tool.Strict, CacheControl: encodeAnthropicCacheControl(tool.Cache)})
	}
	if len(encoded) == 0 {
		return nil
	}
	var err error
	wire.Tools, err = json.Marshal(encoded)
	return err
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
	if role == llmprotocol.RoleDeveloper {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "developer_message_position", "developer messages must be normalized as instructions", nil)
	}
	if role == llmprotocol.RoleTool {
		role = llmprotocol.RoleUser
	}
	if role != llmprotocol.RoleSystem && role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
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
		return anthropicContentWire{Type: "text", Text: content.Text, CacheControl: encodeAnthropicCacheControl(content.Cache)}, nil
	case llmprotocol.ContentReasoning:
		if content.Cache != nil {
			return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_cache_directive", "Anthropic Messages cannot attach cache_control to thinking content", nil)
		}
		return anthropicContentWire{Type: "thinking", Thinking: content.Text, Signature: content.Signature, CacheControl: encodeAnthropicCacheControl(content.Cache)}, nil
	case llmprotocol.ContentImage, llmprotocol.ContentFile:
		if content.Kind == llmprotocol.ContentFile && content.Detail != "" {
			return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_detail", "Anthropic Messages cannot encode file detail", nil)
		}
		block := encodeAnthropicMediaBlock(content)
		block.CacheControl = encodeAnthropicCacheControl(content.Cache)
		return block, nil
	case llmprotocol.ContentToolCall:
		block, err := encodeAnthropicToolCallBlock(content.ToolCall)
		block.CacheControl = encodeAnthropicCacheControl(content.Cache)
		return block, err
	case llmprotocol.ContentToolResult:
		block, err := encodeAnthropicToolResultBlock(content.ToolResult)
		block.CacheControl = encodeAnthropicCacheControl(content.Cache)
		return block, err
	case llmprotocol.ContentRefusal:
		if content.Cache != nil {
			return anthropicContentWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_cache_directive", "Anthropic Messages cannot attach cache_control to refusal content", nil)
		}
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
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Model        string              `json:"model"`
	Content      json.RawMessage     `json:"content"`
	StopReason   *string             `json:"stop_reason"`
	StopSequence *string             `json:"stop_sequence"`
	Usage        *anthropicUsageWire `json:"usage"`
	Error        *anthropicErrorWire `json:"error,omitempty"`
	Container    json.RawMessage     `json:"container"`
	StopDetails  json.RawMessage     `json:"stop_details"`
}

type anthropicUsageWire struct {
	InputTokens              int64                           `json:"input_tokens"`
	OutputTokens             int64                           `json:"output_tokens"`
	CacheCreationInputTokens int64                           `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64                           `json:"cache_read_input_tokens"`
	CacheCreation            anthropicCacheCreationUsageWire `json:"cache_creation"`
	InferenceGeo             string                          `json:"inference_geo"`
	OutputTokensDetails      anthropicOutputUsageDetailsWire `json:"output_tokens_details"`
	ServerToolUse            anthropicServerToolUsageWire    `json:"server_tool_use"`
	ServiceTier              string                          `json:"service_tier"`
}

type anthropicCacheCreationUsageWire struct {
	Ephemeral1hInputTokens int64 `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens int64 `json:"ephemeral_5m_input_tokens"`
}

type anthropicOutputUsageDetailsWire struct {
	ThinkingTokens int64 `json:"thinking_tokens"`
}

type anthropicServerToolUsageWire struct {
	WebFetchRequests  int64 `json:"web_fetch_requests"`
	WebSearchRequests int64 `json:"web_search_requests"`
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
	if err := validateAnthropicResponseResource(wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	diagnostics := anthropicResponseMetadataDiagnostics(wire, policy)
	response, err := decodeAnthropicResponseResource(wire, policy)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	appendAnthropicResponseUsage(&response, wire.Usage, policy, &diagnostics)
	return response, responseEnvelope(llmprotocol.AnthropicMessagesV1, body, response.Generation, response.SourceStopReason, policy), diagnostics, nil
}

func anthropicResponseMetadataDiagnostics(wire anthropicResponseWire, policy llmprotocol.Policy) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	if len(wire.Container) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Container), []byte("null")) {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "container", "container execution metadata is not model output")
	}
	if len(wire.StopDetails) > 0 && !bytes.Equal(bytes.TrimSpace(wire.StopDetails), []byte("null")) {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "stop_details", "structured refusal detail has no neutral representation")
	}
	return diagnostics
}

func decodeAnthropicResponseResource(wire anthropicResponseWire, policy llmprotocol.Policy) (llmprotocol.Response, error) {
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Type), Code: wire.Error.Type, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	contents, err := decodeAnthropicResponseContent(wire.Content, policy)
	if err != nil && wire.Error == nil {
		return llmprotocol.Response{}, err
	}
	if len(contents) > 0 {
		response.Output = []llmprotocol.OutputItem{{ID: llmprotocol.StableID("anthropic-response", wire.ID), Role: llmprotocol.RoleAssistant, Content: contents}}
	}
	if wire.StopReason != nil && response.Error == nil {
		response.SourceStopReason = *wire.StopReason
		response.StopReason = decodeAnthropicStop(*wire.StopReason)
		if response.StopReason == llmprotocol.StopSequence && wire.StopSequence != nil {
			response.MatchedStopSequence = *wire.StopSequence
		}
	}
	return response, nil
}

func appendAnthropicResponseUsage(
	response *llmprotocol.Response,
	usage *anthropicUsageWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) {
	if usage == nil {
		return
	}
	response.Usage = decodeAnthropicUsage(*usage)
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.AnthropicMessagesV1, map[string]bool{
		"usage.cache_creation": usage.CacheCreation.Ephemeral1hInputTokens != 0 ||
			usage.CacheCreation.Ephemeral5mInputTokens != 0,
		"usage.inference_geo": usage.InferenceGeo != "",
		"usage.server_tool_use": usage.ServerToolUse.WebFetchRequests != 0 ||
			usage.ServerToolUse.WebSearchRequests != 0,
		"usage.service_tier": usage.ServiceTier != "",
	}, "provider usage metadata has no neutral accounting bucket")
}

func decodeAnthropicUsage(wire anthropicUsageWire) llmprotocol.Usage {
	inputTotal := wire.InputTokens + wire.CacheReadInputTokens + wire.CacheCreationInputTokens
	reasoning := int64(0)
	reasoning = wire.OutputTokensDetails.ThinkingTokens
	other := wire.OutputTokens - reasoning
	if other < 0 {
		other = 0
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(wire.InputTokens), InputCacheRead: authoritative(wire.CacheReadInputTokens), InputCacheWrite: authoritative(wire.CacheCreationInputTokens),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(inputTotal), OutputTotal: authoritative(wire.OutputTokens), Total: llmprotocol.TokenCount{Value: llmprotocol.Int64(inputTotal + wire.OutputTokens), Provenance: llmprotocol.UsageDerived},
	}
}

func (AnthropicMessagesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if response.Error != nil {
		var diagnostics llmprotocol.Diagnostics
		if response.Usage.State == llmprotocol.UsageAvailable {
			appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1, "usage", "Messages error envelopes cannot carry token usage")
		}
		return encodeAnthropicError(response.Error, response.ProviderRequestID), diagnostics, nil
	}
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if usageUnavailable(response.Usage) {
		if err := appendLossy(
			&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1,
			"usage", "Messages requires usage; emitted an explicit zero-valued usage object",
		); err != nil {
			return nil, diagnostics, err
		}
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
	if response.StopReason == llmprotocol.StopSequence {
		wire.StopSequence = &response.MatchedStopSequence
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeAnthropicUsage(usage llmprotocol.Usage) *anthropicUsageWire {
	wire := newAnthropicUsageWire()
	if usageUnavailable(usage) {
		return wire
	}
	inputTokens := tokenValue(usage.InputUncached)
	if usage.InputUncached.Value == nil {
		inputTokens = tokenValue(usage.InputTotal)
	}
	cacheWrite := tokenValue(usage.InputCacheWrite)
	*wire = anthropicUsageWire{
		InputTokens: inputTokens, OutputTokens: tokenValue(usage.OutputTotal),
		CacheCreationInputTokens: cacheWrite, CacheReadInputTokens: tokenValue(usage.InputCacheRead),
		CacheCreation:       anthropicCacheCreationUsageWire{Ephemeral5mInputTokens: cacheWrite},
		InferenceGeo:        "global",
		OutputTokensDetails: anthropicOutputUsageDetailsWire{ThinkingTokens: tokenValue(usage.OutputReasoning)},
		ServerToolUse:       anthropicServerToolUsageWire{},
		ServiceTier:         "standard",
	}
	return wire
}

func newAnthropicUsageWire() *anthropicUsageWire {
	return &anthropicUsageWire{
		InferenceGeo: "global",
		ServiceTier:  "standard",
	}
}

func usageUnavailable(usage llmprotocol.Usage) bool {
	return usage.State == llmprotocol.UsageUnavailable ||
		usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil
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
	case "pause_turn":
		return llmprotocol.StopPaused
	case "model_context_window_exceeded":
		return llmprotocol.StopContextWindow
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
	case llmprotocol.StopPaused:
		return "pause_turn"
	case llmprotocol.StopContextWindow:
		return "model_context_window_exceeded"
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
	if wire.Type != "error" {
		return llmprotocol.TransportError{}, nil, invalidProviderResponse(
			"invalid_upstream_error_envelope",
			"Anthropic transport error type must be error",
		)
	}
	if wire.Error == nil {
		return llmprotocol.TransportError{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_error_required",
			"upstream transport error body is missing error details",
			nil,
		)
	}
	if err := validateTransportErrorDetails(wire.Error.Type, wire.Error.Message); err != nil {
		return llmprotocol.TransportError{}, nil, err
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
