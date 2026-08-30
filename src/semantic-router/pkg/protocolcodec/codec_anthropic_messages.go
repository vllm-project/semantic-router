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
