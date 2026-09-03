package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
	if len(request.AnthropicContextManagement) > 0 {
		wire.ContextManagement = append(json.RawMessage(nil), request.AnthropicContextManagement...)
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
