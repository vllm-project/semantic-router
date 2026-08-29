package llmprotocol

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"strings"
)

func ValidateRequest(request Request, limits Limits) error {
	blocks, err := validateRequestEnvelope(request, limits)
	if err != nil {
		return err
	}
	if messageErr := validateRequestMessages(request, limits, &blocks); messageErr != nil {
		return messageErr
	}
	if instructionErr := validateRequestInstructions(request.Instructions, limits, &blocks); instructionErr != nil {
		return instructionErr
	}
	if limits.ContentBlocks > 0 && blocks > limits.ContentBlocks {
		return NewError(ErrorInvalidRequest, "content_limit", "content block limit exceeded", nil)
	}
	namedTools, schemaBytes, err := validateRequestTools(request.Tools, limits)
	if err != nil {
		return err
	}
	if err := validateToolChoice(request.ToolChoice, namedTools, len(request.Tools), request.ImageGeneration != nil); err != nil {
		return err
	}
	if err := validateImageGenerationOptions(request.ImageGeneration, limits); err != nil {
		return err
	}
	if err := validateOutputFormat(request.OutputFormat, schemaBytes, limits); err != nil {
		return err
	}
	if err := validateSampling(request.Sampling, limits); err != nil {
		return err
	}
	return validateReasoning(request, limits)
}

func validateRequestEnvelope(request Request, limits Limits) (int, error) {
	if err := validateRequestIdentity(request, limits); err != nil {
		return 0, err
	}
	if err := validateRequestCardinality(request, limits); err != nil {
		return 0, err
	}
	if err := validateCandidateCount(request.CandidateCount, limits.Candidates); err != nil {
		return 0, err
	}
	if !request.Stream && (request.StreamOptions.IncludeUsage != nil || request.StreamOptions.IncludeObfuscation != nil) {
		return 0, NewError(ErrorInvalidRequest, "stream_options_without_stream", "stream options require streaming", nil)
	}
	blocks := 0
	for _, instruction := range request.Instructions {
		blocks += len(instruction.Content)
	}
	if err := validateMetadata(request.Metadata, limits); err != nil {
		return 0, err
	}
	return blocks, nil
}

func validateRequestIdentity(request Request, limits Limits) error {
	if strings.TrimSpace(request.Model) == "" {
		return NewError(ErrorInvalidRequest, "model_required", "model is required", nil)
	}
	if exceeds(request.Model, limits.ModelBytes) {
		return NewError(ErrorInvalidRequest, "model_limit", "model exceeds the configured limit", nil)
	}
	if request.Generation == 0 {
		return NewError(ErrorInternal, "generation_required", "semantic generation is required", nil)
	}
	if exceeds(request.EndUserID, limits.IdentifierBytes) {
		return NewError(ErrorInvalidRequest, "end_user_id_limit", "end-user ID exceeds the configured limit", nil)
	}
	if request.PreviousResponseID != "" && request.ConversationID != "" {
		return NewError(
			ErrorInvalidRequest,
			"conflicting_conversation_state",
			"previous response and conversation references cannot be used together",
			nil,
		)
	}
	if request.Truncation != "" && request.Truncation != "disabled" && request.Truncation != "auto" {
		return NewError(ErrorInvalidRequest, "invalid_truncation", "truncation mode is invalid", nil)
	}
	return nil
}

func validateRequestCardinality(request Request, limits Limits) error {
	if limits.Messages > 0 && len(request.Messages) > limits.Messages {
		return NewError(ErrorInvalidRequest, "messages_limit", "message limit exceeded", nil)
	}
	if limits.Instructions > 0 && len(request.Instructions) > limits.Instructions {
		return NewError(ErrorInvalidRequest, "instructions_limit", "instruction limit exceeded", nil)
	}
	if limits.Tools > 0 && len(request.Tools) > limits.Tools {
		return NewError(ErrorInvalidRequest, "tools_limit", "tool limit exceeded", nil)
	}
	return nil
}

func validateCandidateCount(candidateCount *int64, limit int) error {
	if candidateCount == nil {
		return nil
	}
	if *candidateCount <= 0 {
		return NewError(ErrorInvalidRequest, "invalid_candidate_count", "candidate count must be positive", nil)
	}
	if limit > 0 && *candidateCount > int64(limit) {
		return NewError(ErrorInvalidRequest, "candidate_count_limit", "candidate count exceeds the configured limit", nil)
	}
	return nil
}

func validateMetadata(metadata map[string]string, limits Limits) error {
	metadataBytes := 0
	if limits.MetadataEntries > 0 && len(metadata) > limits.MetadataEntries {
		return NewError(ErrorInvalidRequest, "metadata_entries_limit", "metadata entry limit exceeded", nil)
	}
	for key, value := range metadata {
		if limits.MetadataKeyBytes > 0 && len(key) > limits.MetadataKeyBytes ||
			limits.MetadataValueBytes > 0 && len(value) > limits.MetadataValueBytes {
			return NewError(ErrorInvalidRequest, "metadata_field_limit", "metadata key or value limit exceeded", nil)
		}
		metadataBytes += len(key) + len(value)
	}
	if limits.MetadataBytes > 0 && metadataBytes > limits.MetadataBytes {
		return NewError(ErrorInvalidRequest, "metadata_limit", "metadata limit exceeded", nil)
	}
	return nil
}

func validateRequestMessages(request Request, limits Limits, blocks *int) error {
	toolCalls := make(map[string]struct{})
	toolResults := make(map[string]struct{})
	for _, message := range request.Messages {
		if err := validateRequestMessage(
			message, limits, blocks, toolCalls, toolResults, request.PreviousResponseID != "",
		); err != nil {
			return err
		}
	}
	return nil
}

func validateRequestMessage(
	message Message,
	limits Limits,
	blocks *int,
	toolCalls map[string]struct{},
	toolResults map[string]struct{},
	retainedHistory bool,
) error {
	if exceeds(message.ID, limits.IdentifierBytes) {
		return NewError(ErrorInvalidRequest, "message_id_limit", "message ID exceeds the configured limit", nil)
	}
	if !validRequestRole(message.Role) {
		return NewError(ErrorInvalidRequest, "invalid_role", "message role is invalid", nil)
	}
	*blocks += len(message.Content)
	if len(message.Content) == 0 {
		return NewError(ErrorInvalidRequest, "empty_message", "messages must contain at least one content block", nil)
	}
	if message.Role == RoleTool && len(message.Content) != 1 {
		return NewError(ErrorInvalidRequest, "tool_message_cardinality", "tool messages contain exactly one tool result", nil)
	}
	for _, content := range message.Content {
		if err := validateMessageContent(message.Role, content, limits, blocks); err != nil {
			return err
		}
		if err := recordToolLink(content, toolCalls, toolResults, retainedHistory); err != nil {
			return err
		}
	}
	return nil
}

func validRequestRole(role Role) bool {
	return role == RoleSystem || role == RoleDeveloper || role == RoleUser ||
		role == RoleAssistant || role == RoleTool
}

func validateMessageContent(role Role, content Content, limits Limits, blocks *int) error {
	if err := validateRoleContent(role, content); err != nil {
		return err
	}
	return validateContent(content, blocks, limits, 0)
}

func validateRequestInstructions(instructions []InstructionBlock, limits Limits, blocks *int) error {
	for _, instruction := range instructions {
		if instruction.Role != RoleSystem && instruction.Role != RoleDeveloper {
			return NewError(ErrorInvalidRequest, "invalid_instruction_role", "instruction role must be system or developer", nil)
		}
		if len(instruction.Content) == 0 {
			return NewError(ErrorInvalidRequest, "empty_instruction", "instructions must contain at least one content block", nil)
		}
		for _, content := range instruction.Content {
			if content.Kind == ContentToolCall || content.Kind == ContentToolResult {
				return NewError(ErrorInvalidRequest, "invalid_instruction_content", "instructions cannot contain tool control blocks", nil)
			}
			if err := validateContent(content, blocks, limits, 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateRequestTools(tools []Tool, limits Limits) (map[string]struct{}, int, error) {
	namedTools := make(map[string]struct{}, len(tools))
	schemaBytes := 0
	for _, tool := range tools {
		if err := validateRequestTool(tool, limits); err != nil {
			return nil, 0, err
		}
		if _, duplicate := namedTools[tool.Name]; duplicate {
			return nil, 0, NewError(ErrorInvalidRequest, "duplicate_tool", "tool names must be unique", nil)
		}
		schemaBytes += len(tool.InputSchema)
		if limits.SchemaBytes > 0 && schemaBytes > limits.SchemaBytes {
			return nil, 0, NewError(ErrorInvalidRequest, "schema_limit", "total schema limit exceeded", nil)
		}
		namedTools[tool.Name] = struct{}{}
	}
	return namedTools, schemaBytes, nil
}

func validateRequestTool(tool Tool, limits Limits) error {
	if strings.TrimSpace(tool.Name) == "" || len(tool.InputSchema) == 0 || !json.Valid(tool.InputSchema) {
		return NewError(ErrorInvalidRequest, "invalid_tool", "tool name and JSON Schema are required", nil)
	}
	if exceeds(tool.Name, limits.ToolNameBytes) || exceeds(tool.Description, limits.ToolDescriptionBytes) {
		return NewError(ErrorInvalidRequest, "tool_text_limit", "tool name or description exceeds the configured limit", nil)
	}
	if limits.SchemaBytes > 0 && len(tool.InputSchema) > limits.SchemaBytes {
		return NewError(ErrorInvalidRequest, "schema_limit", "tool schema limit exceeded", nil)
	}
	if err := validateCacheDirective(tool.Cache); err != nil {
		return err
	}
	return validateSchemaObject(tool.InputSchema, "tool schema", limits)
}

func validateToolChoice(choice ToolChoice, namedTools map[string]struct{}, toolCount int, hasImageGeneration bool) error {
	if !validToolChoiceMode(choice.Mode) {
		return NewError(ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
	}
	if choice.Mode == ToolChoiceNamed {
		return validateNamedToolChoice(choice.Name, namedTools)
	}
	if choice.Name != "" {
		return NewError(ErrorInvalidRequest, "invalid_tool_choice", "only named tool choice may contain a name", nil)
	}
	if choice.Mode == ToolChoiceImageGeneration && !hasImageGeneration {
		return NewError(ErrorInvalidRequest, "image_generation_tool_required", "image-generation tool choice requires a declared image-generation tool", nil)
	}
	if choice.Mode == ToolChoiceRequired && toolCount == 0 && !hasImageGeneration {
		return NewError(ErrorInvalidRequest, "tools_required", "tool choice requires at least one declared tool", nil)
	}
	return nil
}

func validToolChoiceMode(mode ToolChoiceMode) bool {
	switch mode {
	case "", ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceNamed, ToolChoiceImageGeneration:
		return true
	default:
		return false
	}
}

func validateNamedToolChoice(name string, namedTools map[string]struct{}) error {
	if strings.TrimSpace(name) == "" {
		return NewError(ErrorInvalidRequest, "tool_choice_name_required", "named tool choice requires a name", nil)
	}
	if _, found := namedTools[name]; !found {
		return NewError(ErrorInvalidRequest, "unknown_tool_choice", "named tool choice does not reference a declared tool", nil)
	}
	return nil
}

func validateOutputFormat(format OutputFormat, schemaBytes int, limits Limits) error {
	switch format.Kind {
	case "", OutputText, OutputJSONObject, OutputJSONSchema:
	default:
		return NewError(ErrorInvalidRequest, "invalid_output_format", "output format is invalid", nil)
	}
	if format.Kind != OutputJSONSchema {
		if !outputFormatHasSchemaFields(format) {
			return nil
		}
		return NewError(ErrorInvalidRequest, "invalid_output_format", "output format contains fields from another format kind", nil)
	}
	return validateJSONSchemaOutputFormat(format, schemaBytes, limits)
}

func outputFormatHasSchemaFields(format OutputFormat) bool {
	return len(format.Schema) > 0 || format.Name != "" || format.Description != "" || format.Strict != nil
}

func validateJSONSchemaOutputFormat(format OutputFormat, schemaBytes int, limits Limits) error {
	if len(format.Schema) == 0 || !json.Valid(format.Schema) {
		return NewError(ErrorInvalidRequest, "invalid_output_schema", "output JSON Schema is invalid", nil)
	}
	if limits.SchemaBytes > 0 && len(format.Schema) > limits.SchemaBytes {
		return NewError(ErrorInvalidRequest, "schema_limit", "output schema limit exceeded", nil)
	}
	if limits.SchemaBytes > 0 && schemaBytes+len(format.Schema) > limits.SchemaBytes {
		return NewError(ErrorInvalidRequest, "schema_limit", "total schema limit exceeded", nil)
	}
	if err := validateSchemaObject(format.Schema, "output schema", limits); err != nil {
		return err
	}
	if strings.TrimSpace(format.Name) == "" {
		return NewError(ErrorInvalidRequest, "output_schema_name_required", "output JSON Schema requires a name", nil)
	}
	return nil
}

func validateSchemaObject(schema json.RawMessage, label string, limits Limits) error {
	if err := ValidateJSONObject(schema, limits.JSONDepth); err != nil {
		return NewError(ErrorInvalidRequest, "invalid_schema", label+" must be a JSON object", err)
	}
	return nil
}

func validateSampling(sampling Sampling, limits Limits) error {
	if err := validateSamplingScalars(sampling); err != nil {
		return err
	}
	return validateStopSequences(sampling.Stop, limits)
}

func validateSamplingScalars(sampling Sampling) error {
	if err := validateSamplingProbability(sampling); err != nil {
		return err
	}
	if err := validateSamplingCounts(sampling); err != nil {
		return err
	}
	for _, penalty := range []*float64{sampling.FrequencyPenalty, sampling.PresencePenalty} {
		if penalty != nil && (!finiteFloat(*penalty) || *penalty < -2 || *penalty > 2) {
			return NewError(ErrorInvalidRequest, "invalid_penalty", "sampling penalty must be between -2 and 2", nil)
		}
	}
	return nil
}

func validateSamplingProbability(sampling Sampling) error {
	if sampling.Temperature != nil && (!finiteFloat(*sampling.Temperature) || *sampling.Temperature < 0 || *sampling.Temperature > 2) {
		return NewError(ErrorInvalidRequest, "invalid_temperature", "temperature must be between 0 and 2", nil)
	}
	if sampling.TopP != nil && (!finiteFloat(*sampling.TopP) || *sampling.TopP < 0 || *sampling.TopP > 1) {
		return NewError(ErrorInvalidRequest, "invalid_top_p", "top_p must be between 0 and 1", nil)
	}
	return nil
}

func finiteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateSamplingCounts(sampling Sampling) error {
	if sampling.TopK != nil && *sampling.TopK < 0 {
		return NewError(ErrorInvalidRequest, "invalid_top_k", "top_k cannot be negative", nil)
	}
	if sampling.MaxOutputTokens != nil && *sampling.MaxOutputTokens < 0 {
		return NewError(ErrorInvalidRequest, "invalid_max_output_tokens", "max output tokens cannot be negative", nil)
	}
	return nil
}

func validateStopSequences(stops []string, limits Limits) error {
	if limits.StopSequences > 0 && len(stops) > limits.StopSequences {
		return NewError(ErrorInvalidRequest, "stop_limit", "too many stop sequences", nil)
	}
	stopBytes := 0
	for _, stop := range stops {
		if stop == "" {
			return NewError(ErrorInvalidRequest, "invalid_stop", "stop sequences must be non-empty and bounded", nil)
		}
		stopBytes += len(stop)
		if limits.StopBytes > 0 && stopBytes > limits.StopBytes {
			return NewError(ErrorInvalidRequest, "stop_bytes_limit", "stop sequences exceed the configured limit", nil)
		}
	}
	return nil
}

func validateReasoning(request Request, limits Limits) error {
	if err := validateReasoningEffort(request, limits); err != nil {
		return err
	}
	if err := validateReasoningDisplay(request); err != nil {
		return err
	}
	return validateReasoningMode(request)
}

func validateReasoningEffort(request Request, limits Limits) error {
	if exceeds(request.ReasoningEffort, limits.ReasoningEffortBytes) {
		return NewError(ErrorInvalidRequest, "reasoning_effort_limit", "reasoning effort exceeds the configured limit", nil)
	}
	switch request.ReasoningEffort {
	case "", "none", "minimal", "low", "medium", "high", "xhigh", "max":
	default:
		return NewError(ErrorInvalidRequest, "invalid_reasoning_effort", "reasoning effort is invalid", nil)
	}
	if request.ReasoningBudgetTokens != nil && *request.ReasoningBudgetTokens <= 0 {
		return NewError(ErrorInvalidRequest, "invalid_reasoning_budget", "reasoning budget must be positive", nil)
	}
	return nil
}

func validateReasoningDisplay(request Request) error {
	switch request.ReasoningDisplay {
	case "", "summarized", "omitted":
	default:
		return NewError(ErrorInvalidRequest, "invalid_reasoning_display", "reasoning display mode is invalid", nil)
	}
	if request.ReasoningDisplay != "" &&
		request.ReasoningMode != ReasoningModeEnabled &&
		request.ReasoningMode != ReasoningModeAdaptive {
		return NewError(ErrorInvalidRequest, "conflicting_reasoning_display", "reasoning display requires enabled or adaptive reasoning", nil)
	}
	return nil
}

func validateReasoningMode(request Request) error {
	switch request.ReasoningMode {
	case "", ReasoningModeEnabled:
	case ReasoningModeDisabled, ReasoningModeAdaptive:
		if request.ReasoningBudgetTokens != nil || strings.TrimSpace(request.ReasoningEffort) != "" {
			return NewError(ErrorInvalidRequest, "conflicting_reasoning_control", string(request.ReasoningMode)+" reasoning cannot include an effort or token budget", nil)
		}
	default:
		return NewError(ErrorInvalidRequest, "invalid_reasoning_mode", "reasoning mode is invalid", nil)
	}
	if request.ReasoningMode == ReasoningModeEnabled && request.ReasoningBudgetTokens == nil {
		return NewError(ErrorInvalidRequest, "reasoning_budget_required", "enabled reasoning requires a token budget", nil)
	}
	return nil
}

func recordToolLink(content Content, calls, results map[string]struct{}, retainedHistory bool) error {
	switch {
	case content.Kind == ContentToolCall && content.ToolCall != nil:
		return recordToolCall(content.ToolCall, calls, results)
	case content.Kind == ContentToolResult && content.ToolResult != nil:
		return recordToolResult(content.ToolResult, calls, results, retainedHistory)
	default:
		return nil
	}
}

func recordToolCall(call *ToolCall, calls, results map[string]struct{}) error {
	if _, resultAlreadySeen := results[call.ID]; resultAlreadySeen {
		return NewError(ErrorInvalidRequest, "tool_result_order", "tool result must follow its request tool call", nil)
	}
	if _, duplicate := calls[call.ID]; duplicate {
		return NewError(ErrorInvalidRequest, "duplicate_tool_call", "tool call IDs must be unique", nil)
	}
	calls[call.ID] = struct{}{}
	return nil
}

func recordToolResult(
	result *ToolResult,
	calls map[string]struct{},
	results map[string]struct{},
	retainedHistory bool,
) error {
	if _, duplicate := results[result.CallID]; duplicate {
		return NewError(ErrorInvalidRequest, "duplicate_tool_result", "tool results must be unique per call", nil)
	}
	_, found := calls[result.CallID]
	if found && result.DeferredLink {
		return NewError(ErrorInvalidRequest, "invalid_deferred_tool_result", "a locally linked tool result cannot be deferred", nil)
	}
	if !found && (!retainedHistory || !result.DeferredLink) {
		return NewError(ErrorInvalidRequest, "orphan_tool_result", "tool result must follow its request tool call or carry a retained-history link", nil)
	}
	results[result.CallID] = struct{}{}
	return nil
}

// MarkDeferredToolLinks reconciles tool-result links against the messages
// currently materialized in the request. A missing call is deferred only while
// previous_response_id still identifies retained history; after that history is
// expanded locally, ordinary lifecycle validation applies to every result.
func MarkDeferredToolLinks(request *Request) {
	if request == nil {
		return
	}
	calls := make(map[string]struct{})
	for messageIndex := range request.Messages {
		for contentIndex := range request.Messages[messageIndex].Content {
			content := &request.Messages[messageIndex].Content[contentIndex]
			if content.Kind == ContentToolCall && content.ToolCall != nil {
				calls[content.ToolCall.ID] = struct{}{}
				continue
			}
			if content.Kind != ContentToolResult || content.ToolResult == nil {
				continue
			}
			_, local := calls[content.ToolResult.CallID]
			content.ToolResult.DeferredLink = request.PreviousResponseID != "" && !local
		}
	}
}

func validateRoleContent(role Role, content Content) error {
	switch role {
	case RoleSystem, RoleDeveloper:
		if contentKindIsOneOf(content.Kind, ContentToolCall, ContentToolResult, ContentGeneratedImage) {
			return NewError(ErrorInvalidRequest, "invalid_role_content", "system and developer messages cannot contain tool control blocks", nil)
		}
	case RoleAssistant:
		if content.Kind == ContentToolResult {
			return NewError(ErrorInvalidRequest, "invalid_role_content", "assistant messages cannot contain tool results", nil)
		}
	case RoleTool:
		if content.Kind != ContentToolResult {
			return NewError(ErrorInvalidRequest, "invalid_role_content", "tool messages may contain only tool results", nil)
		}
	case RoleUser:
		if contentKindIsOneOf(content.Kind, ContentToolCall, ContentToolResult, ContentRefusal, ContentGeneratedImage) {
			return NewError(ErrorInvalidRequest, "invalid_role_content", "user messages contain assistant-only content", nil)
		}
	}
	return nil
}

func contentKindIsOneOf(kind ContentKind, candidates ...ContentKind) bool {
	for _, candidate := range candidates {
		if kind == candidate {
			return true
		}
	}
	return false
}

func validateContent(content Content, blocks *int, limits Limits, depth int) error {
	if limits.ToolResultDepth <= 0 || depth > limits.ToolResultDepth {
		return NewError(ErrorInvalidRequest, "tool_result_depth", "nested tool result depth exceeded", nil)
	}
	if err := validateCacheDirective(content.Cache); err != nil {
		return err
	}
	switch content.Kind {
	case ContentText, ContentRefusal, ContentReasoning:
		return validateTextContent(content, limits)
	case ContentImage, ContentAudio, ContentVideo, ContentFile:
		return validateMediaContent(content, limits)
	case ContentToolCall:
		return validateToolCallContent(content, limits)
	case ContentToolResult:
		return validateToolResultContent(content, blocks, limits, depth)
	case ContentGeneratedImage:
		return validateGeneratedImageContent(content, limits)
	default:
		return NewError(ErrorInvalidRequest, "unknown_content_kind", "content kind is unsupported", nil)
	}
}

func validateCacheDirective(cache *CacheDirective) error {
	if cache == nil {
		return nil
	}
	if cache.Type != "ephemeral" {
		return NewError(ErrorInvalidRequest, "invalid_cache_directive", "cache directive type must be ephemeral", nil)
	}
	switch cache.TTL {
	case "", "5m", "1h":
		return nil
	default:
		return NewError(ErrorInvalidRequest, "invalid_cache_directive", "cache directive TTL must be 5m or 1h", nil)
	}
}

func validateTextContent(content Content, limits Limits) error {
	if hasNonTextFields(content) {
		return NewError(ErrorInvalidRequest, "invalid_content", "text content contains fields from another content kind", nil)
	}
	if content.Kind == ContentReasoning {
		switch content.Reasoning {
		case "", ReasoningScopeText, ReasoningScopeSummary:
		default:
			return NewError(ErrorInvalidRequest, "invalid_reasoning_scope", "reasoning content scope is unsupported", nil)
		}
		if content.Reasoning == ReasoningScopeSummary && content.Signature != "" {
			return NewError(ErrorInvalidRequest, "invalid_reasoning_signature", "reasoning summaries cannot carry a private reasoning signature", nil)
		}
	}
	if content.Kind != ContentText && len(content.Citations) != 0 {
		return NewError(ErrorInvalidRequest, "invalid_content", "only text content may contain citations", nil)
	}
	return ValidateTextCitations(content.Text, content.Citations, limits)
}

func hasNonTextFields(content Content) bool {
	return content.ToolCall != nil || content.ToolResult != nil || content.GeneratedImage != nil || content.URL != "" || content.Data != "" ||
		content.FileID != "" || content.Filename != "" || content.MediaType != "" || content.Detail != "" ||
		(content.Kind != ContentReasoning && (content.Signature != "" || content.Reasoning != ""))
}

func validateMediaContent(content Content, limits Limits) error {
	if err := validateMediaBounds(content, limits); err != nil {
		return err
	}
	if content.ToolCall != nil || content.ToolResult != nil || content.GeneratedImage != nil || content.Text != "" ||
		content.Signature != "" || len(content.Citations) != 0 {
		return NewError(ErrorInvalidRequest, "invalid_content", "media content contains fields from another content kind", nil)
	}
	if mediaSourceCount(content) != 1 {
		return NewError(ErrorInvalidRequest, "media_reference_required", "media content requires exactly one data source or reference", nil)
	}
	if err := validateMediaSource(content); err != nil {
		return err
	}
	return validateMediaKindFields(content)
}

func validateMediaKindFields(content Content) error {
	if content.Kind != ContentImage && content.Kind != ContentFile && content.Detail != "" {
		return NewError(ErrorInvalidRequest, "invalid_content", "only image or file content may specify detail", nil)
	}
	if content.Kind != ContentFile && content.Filename != "" {
		return NewError(ErrorInvalidRequest, "invalid_content", "only file content may specify a filename", nil)
	}
	return nil
}

func validateMediaBounds(content Content, limits Limits) error {
	if limits.MediaDataBytes > 0 && len(content.Data) > limits.MediaDataBytes {
		return NewError(ErrorInvalidRequest, "media_data_limit", "inline media exceeds the configured limit", nil)
	}
	for _, reference := range []string{content.URL, content.FileID, content.Filename, content.MediaType, content.Detail} {
		if limits.MediaReferenceBytes > 0 && len(reference) > limits.MediaReferenceBytes {
			return NewError(ErrorInvalidRequest, "media_reference_limit", "media reference exceeds the configured limit", nil)
		}
	}
	return nil
}

func mediaSourceCount(content Content) int {
	sources := 0
	for _, source := range []string{content.URL, content.Data, content.FileID} {
		if source != "" {
			sources++
		}
	}
	return sources
}

func validateMediaSource(content Content) error {
	if content.URL != "" {
		return validateMediaURL(content.URL)
	}
	if content.Data == "" {
		return nil
	}
	if strings.TrimSpace(content.MediaType) == "" {
		return NewError(ErrorInvalidRequest, "media_type_required", "inline media requires a media type", nil)
	}
	decoder := base64.NewDecoder(base64.StdEncoding.Strict(), strings.NewReader(content.Data))
	if _, err := io.Copy(io.Discard, decoder); err != nil {
		return NewError(ErrorInvalidRequest, "invalid_media_data", "inline media must be valid base64", err)
	}
	return nil
}

func validateToolCallContent(content Content, limits Limits) error {
	call := content.ToolCall
	if call == nil || strings.TrimSpace(call.ID) == "" ||
		strings.TrimSpace(call.Name) == "" {
		return NewError(ErrorInvalidRequest, "invalid_tool_call", "tool call requires an ID, name, and JSON arguments", nil)
	}
	if err := ValidateJSONObject([]byte(call.Arguments), limits.JSONDepth); err != nil {
		return NewError(ErrorInvalidRequest, "invalid_tool_call", "tool call arguments must be one strict JSON object", err)
	}
	if exceeds(call.ID, limits.IdentifierBytes) || exceeds(call.Name, limits.ToolNameBytes) {
		return NewError(ErrorInvalidRequest, "tool_call_limit", "tool call ID or name exceeds the configured limit", nil)
	}
	if limits.ToolArgumentsBytes > 0 && len(call.Arguments) > limits.ToolArgumentsBytes {
		return NewError(ErrorInvalidRequest, "tool_arguments_limit", "tool arguments exceed the configured limit", nil)
	}
	if hasToolControlForeignFields(content) || content.ToolResult != nil {
		return NewError(ErrorInvalidRequest, "invalid_content", "tool call contains fields from another content kind", nil)
	}
	return nil
}

func validateToolResultContent(content Content, blocks *int, limits Limits, depth int) error {
	result := content.ToolResult
	if result == nil || strings.TrimSpace(result.CallID) == "" {
		return NewError(ErrorInvalidRequest, "invalid_tool_result", "tool result requires a call ID", nil)
	}
	if exceeds(result.CallID, limits.IdentifierBytes) {
		return NewError(ErrorInvalidRequest, "tool_result_id_limit", "tool result call ID exceeds the configured limit", nil)
	}
	if hasToolControlForeignFields(content) || content.ToolCall != nil {
		return NewError(ErrorInvalidRequest, "invalid_content", "tool result contains fields from another content kind", nil)
	}
	return validateNestedToolResult(result.Content, blocks, limits, depth)
}

func hasToolControlForeignFields(content Content) bool {
	return content.Text != "" || content.URL != "" || content.Data != "" || len(content.Citations) != 0 ||
		content.FileID != "" || content.MediaType != "" || content.Filename != "" ||
		content.Detail != "" || content.Signature != "" || content.GeneratedImage != nil
}

func validateNestedToolResult(contents []Content, blocks *int, limits Limits, depth int) error {
	*blocks += len(contents)
	if limits.ContentBlocks > 0 && *blocks > limits.ContentBlocks {
		return NewError(ErrorInvalidRequest, "content_limit", "content block limit exceeded", nil)
	}
	for _, nested := range contents {
		if nested.Kind == ContentToolResult || nested.Kind == ContentToolCall {
			return NewError(ErrorInvalidRequest, "nested_tool_control", "tool results cannot contain tool control blocks", nil)
		}
		if err := validateContent(nested, blocks, limits, depth+1); err != nil {
			return err
		}
	}
	return nil
}
