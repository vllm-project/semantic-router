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
	Thinking *anthropicThinkingWire `json:"thinking,omitempty"`
	Stream   bool                   `json:"stream,omitempty"`
}

type anthropicMessageWire struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicContentWire struct {
	Type      string                    `json:"type"`
	Text      string                    `json:"text,omitempty"`
	Thinking  string                    `json:"thinking,omitempty"`
	Signature string                    `json:"signature,omitempty"`
	ID        string                    `json:"id,omitempty"`
	Name      string                    `json:"name,omitempty"`
	Input     json.RawMessage           `json:"input,omitempty"`
	ToolUseID string                    `json:"tool_use_id,omitempty"`
	Content   json.RawMessage           `json:"content,omitempty"`
	IsError   *bool                     `json:"is_error,omitempty"`
	Source    *anthropicMediaSourceWire `json:"source,omitempty"`
}

type anthropicMediaSourceWire struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicToolWire struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      *bool           `json:"strict,omitempty"`
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
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream,
		Sampling: llmprotocol.Sampling{
			Temperature: wire.Temperature, TopP: wire.TopP, TopK: wire.TopK,
			MaxOutputTokens: llmprotocol.Int64(wire.MaxTokens), Stop: append([]string(nil), wire.StopSequences...),
		},
		Trusted: llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.AnthropicMessagesV1},
	}
	if wire.Metadata != nil && wire.Metadata.UserID != "" {
		request.Metadata = map[string]string{"user_id": wire.Metadata.UserID}
	}
	if wire.Thinking != nil && wire.Thinking.Type == "enabled" {
		request.ReasoningBudgetTokens = llmprotocol.Int64(wire.Thinking.BudgetTokens)
	}
	if len(wire.System) > 0 && !bytes.Equal(bytes.TrimSpace(wire.System), []byte("null")) {
		contents, err := decodeAnthropicContent(wire.System, policy)
		if err != nil {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
		}
		request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: contents}}
	}
	for index, messageWire := range wire.Messages {
		message, err := decodeAnthropicMessage(messageWire, index, policy)
		if err != nil {
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
		}
		request.Messages = append(request.Messages, message...)
	}
	for _, toolWire := range wire.Tools {
		schema := toolWire.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{Name: toolWire.Name, Description: toolWire.Description, InputSchema: schema, Strict: toolWire.Strict})
	}
	if wire.ToolChoice != nil {
		switch wire.ToolChoice.Type {
		case "auto":
			request.ToolChoice.Mode = llmprotocol.ToolChoiceAuto
		case "none":
			request.ToolChoice.Mode = llmprotocol.ToolChoiceNone
		case "any":
			request.ToolChoice.Mode = llmprotocol.ToolChoiceRequired
		case "tool":
			request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: wire.ToolChoice.Name}
		default:
			return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "Anthropic tool choice is invalid", nil)
		}
		if wire.ToolChoice.DisableParallelToolUse != nil {
			parallel := !*wire.ToolChoice.DisableParallelToolUse
			request.ParallelToolCalls = &parallel
		}
	}
	return request, requestEnvelope(llmprotocol.AnthropicMessagesV1, body, request.Generation, policy), nil, nil
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
	var blocks []anthropicContentWire
	if err := decodeWire(raw, &blocks, policy); err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: block.Text})
		case "thinking":
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: block.Thinking, Signature: block.Signature})
		case "image", "document":
			if block.Source == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "media_source_required", "Anthropic media source is required", nil)
			}
			kind := llmprotocol.ContentImage
			if block.Type == "document" {
				kind = llmprotocol.ContentFile
			}
			contents = append(contents, llmprotocol.Content{Kind: kind, MediaType: block.Source.MediaType, Data: block.Source.Data, URL: block.Source.URL})
		case "tool_use":
			arguments := string(block.Input)
			if len(block.Input) == 0 {
				arguments = `{}`
			}
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: block.ID, Name: block.Name, Arguments: arguments}})
		case "tool_result":
			resultContent, err := decodeAnthropicContent(block.Content, policy)
			if err != nil {
				return nil, err
			}
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: block.ToolUseID, Content: resultContent, IsError: block.IsError}})
		case "redacted_thinking":
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning", "redacted reasoning cannot be translated", nil)
		default:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic content type is unsupported", nil)
		}
	}
	return contents, nil
}

func (AnthropicMessagesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	for _, instruction := range request.Instructions {
		contentDiagnostics, err := anthropicContentDiagnostics(instruction.Content, request.Trusted.SourceFormat, policy)
		diagnostics = appendDiagnostics(diagnostics, contentDiagnostics, policy.Limits.Diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
	}
	for _, message := range request.Messages {
		contentDiagnostics, err := anthropicContentDiagnostics(message.Content, request.Trusted.SourceFormat, policy)
		diagnostics = appendDiagnostics(diagnostics, contentDiagnostics, policy.Limits.Diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
	}
	if request.PreviousResponseID != "" || request.ConversationID != "" || request.Store != nil {
		if err := appendLossy(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "conversation_state", "Messages has no stateful response reference"); err != nil {
			return nil, diagnostics, err
		}
	}
	if request.OutputFormat.Kind == llmprotocol.OutputJSONObject || request.OutputFormat.Kind == llmprotocol.OutputJSONSchema {
		if err := appendLossy(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "output_format", "Messages has no native strict output format"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := anthropicRequestWire{
		Model: request.Model, Stream: request.Stream, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, TopK: request.Sampling.TopK, StopSequences: append([]string(nil), request.Sampling.Stop...),
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens <= 0 {
		return nil, diagnostics, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "max_tokens_required", "Anthropic Messages requires max output tokens", nil)
	}
	wire.MaxTokens = *request.Sampling.MaxOutputTokens
	if request.ReasoningBudgetTokens != nil {
		wire.Thinking = &anthropicThinkingWire{Type: "enabled", BudgetTokens: *request.ReasoningBudgetTokens}
	}
	if userID := request.Metadata["user_id"]; userID != "" {
		wire.Metadata = &struct {
			UserID string `json:"user_id,omitempty"`
		}{UserID: userID}
	}
	if len(request.Instructions) > 0 {
		contents := make([]llmprotocol.Content, 0)
		for _, instruction := range request.Instructions {
			if instruction.Role == llmprotocol.RoleDeveloper {
				if err := appendLossy(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.AnthropicMessagesV1, "instructions.role", "Messages cannot preserve developer authority"); err != nil {
					return nil, diagnostics, err
				}
			}
			contents = append(contents, instruction.Content...)
		}
		encoded, err := encodeAnthropicContent(contents)
		if err != nil {
			return nil, diagnostics, err
		}
		wire.System = encoded
	}
	for _, message := range request.Messages {
		encoded, err := encodeAnthropicMessage(message)
		if err != nil {
			return nil, diagnostics, err
		}
		wire.Messages = append(wire.Messages, encoded...)
	}
	for _, tool := range request.Tools {
		wire.Tools = append(wire.Tools, anthropicToolWire{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, Strict: tool.Strict})
	}
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
			return nil, diagnostics, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
		}
		if request.ParallelToolCalls != nil {
			disable := !*request.ParallelToolCalls
			wire.ToolChoice.DisableParallelToolUse = &disable
		}
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
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
		switch content.Kind {
		case llmprotocol.ContentText:
			blocks = append(blocks, anthropicContentWire{Type: "text", Text: content.Text})
		case llmprotocol.ContentReasoning:
			blocks = append(blocks, anthropicContentWire{Type: "thinking", Thinking: content.Text, Signature: content.Signature})
		case llmprotocol.ContentImage, llmprotocol.ContentFile:
			typeName := "image"
			if content.Kind == llmprotocol.ContentFile {
				typeName = "document"
			}
			sourceType := "base64"
			if content.URL != "" {
				sourceType = "url"
			}
			blocks = append(blocks, anthropicContentWire{Type: typeName, Source: &anthropicMediaSourceWire{Type: sourceType, MediaType: content.MediaType, Data: content.Data, URL: content.URL}})
		case llmprotocol.ContentToolCall:
			if content.ToolCall == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call is invalid", nil)
			}
			arguments := json.RawMessage(content.ToolCall.Arguments)
			if !json.Valid(arguments) {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_arguments", "tool arguments must be JSON", nil)
			}
			blocks = append(blocks, anthropicContentWire{Type: "tool_use", ID: content.ToolCall.ID, Name: content.ToolCall.Name, Input: arguments})
		case llmprotocol.ContentToolResult:
			if content.ToolResult == nil {
				return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result is invalid", nil)
			}
			result, err := encodeAnthropicContent(content.ToolResult.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, anthropicContentWire{Type: "tool_result", ToolUseID: content.ToolResult.CallID, Content: result, IsError: content.ToolResult.IsError})
		case llmprotocol.ContentRefusal:
			blocks = append(blocks, anthropicContentWire{Type: "text", Text: content.Text})
		default:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as Anthropic Messages", nil)
		}
	}
	return json.Marshal(blocks)
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
}

type anthropicUsageWire struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens,omitempty"`
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
	}
	return response, responseEnvelope(llmprotocol.AnthropicMessagesV1, body, response.Generation, response.SourceStopReason, policy), nil, nil
}

func decodeAnthropicUsage(wire anthropicUsageWire) llmprotocol.Usage {
	uncached := wire.InputTokens - wire.CacheReadInputTokens - wire.CacheCreationInputTokens
	if uncached < 0 {
		uncached = 0
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(uncached), InputCacheRead: authoritative(wire.CacheReadInputTokens), InputCacheWrite: authoritative(wire.CacheCreationInputTokens),
		OutputReasoning: unknownCount(), OutputOther: authoritative(wire.OutputTokens),
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
	switch code {
	case "invalid_request_error", "request_too_large":
		return category == llmprotocol.ErrorInvalidRequest || category == llmprotocol.ErrorUnsupportedFeature
	case "authentication_error":
		return category == llmprotocol.ErrorAuthentication
	case "billing_error":
		return category == llmprotocol.ErrorUpstreamUnavailable
	case "permission_error":
		return category == llmprotocol.ErrorPermission
	case "not_found_error":
		return category == llmprotocol.ErrorNotFound
	case "conflict_error":
		return category == llmprotocol.ErrorConflict
	case "rate_limit_error":
		return category == llmprotocol.ErrorRateLimited
	case "timeout_error":
		return category == llmprotocol.ErrorUpstreamTimeout
	case "api_error", "overloaded_error":
		return category == llmprotocol.ErrorUpstreamUnavailable || category == llmprotocol.ErrorInternal
	default:
		return false
	}
}
