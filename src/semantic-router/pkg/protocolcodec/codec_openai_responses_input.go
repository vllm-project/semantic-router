package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
	"output", "phase", "result", "role", "status", "summary", "type",
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
		return itemType == "message" || itemType == "function_call" || itemType == "reasoning" || itemType == "image_generation_call"
	}
	switch itemType {
	case "message", "function_call", "function_call_output", "reasoning", "item_reference", "image_generation_call":
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
	if itemType == "image_generation_call" {
		for _, field := range []string{"id", "status", "result"} {
			if _, present := object[field]; !present {
				return responsesItemDecodeError(providerOutput, "image generation item is missing required field: "+field, nil)
			}
		}
		for _, field := range []string{"id", "status"} {
			if bytes.Equal(bytes.TrimSpace(object[field]), []byte("null")) {
				return responsesItemDecodeError(providerOutput, "image generation item field cannot be null: "+field, nil)
			}
		}
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
	case "image_generation_call":
		return []string{"id", "result", "status", "type"}
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
	if err := validateResponsesInputItemMetadata(item); err != nil {
		return err
	}
	return decodeResponsesInputItemKind(item, index, request, policy)
}

func validateResponsesInputItemMetadata(item responsesItemWire) error {
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
	if item.Status != "" && item.Type != "image_generation_call" {
		return rejectUnsupportedRequestField("input.status", json.RawMessage(`true`))
	}
	return nil
}

func decodeResponsesInputItemKind(
	item responsesItemWire,
	index int,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
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
	case "image_generation_call":
		request.Messages = append(request.Messages, llmprotocol.Message{
			ID: item.ID, Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind:           llmprotocol.ContentGeneratedImage,
				GeneratedImage: decodeResponsesGeneratedImage(item),
			}},
		})
		return nil
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
	if choice, found := decodeResponsesStringToolChoice(raw); found {
		return choice, nil
	}
	return decodeResponsesObjectToolChoice(raw, policy)
}

func decodeResponsesStringToolChoice(raw json.RawMessage) (llmprotocol.ToolChoice, bool) {
	var mode string
	if json.Unmarshal(raw, &mode) != nil {
		return llmprotocol.ToolChoice{}, false
	}
	switch mode {
	case "auto":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}, true
	case "none":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}, true
	case "required":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceRequired}, true
	default:
		return llmprotocol.ToolChoice{}, false
	}
}

func decodeResponsesObjectToolChoice(raw json.RawMessage, policy llmprotocol.Policy) (llmprotocol.ToolChoice, error) {
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
	if discriminator.Type == "image_generation" {
		var imageChoice struct {
			Type string `json:"type"`
		}
		if err := decodeWireValue(raw, &imageChoice, policy); err != nil {
			return llmprotocol.ToolChoice{}, err
		}
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}, nil
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
		"computer_use_preview", "custom", "file_search", "mcp",
		"programmatic_tool_calling", "shell", "web_search_preview", "web_search_preview_2025_03_11":
		return true
	default:
		return false
	}
}
