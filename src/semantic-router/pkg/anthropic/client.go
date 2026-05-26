// Package anthropic provides transformation functions between OpenAI and Anthropic API formats.
// Used for Envoy-routed requests where the router transforms request/response bodies.
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxTokens is the default max tokens if not specified in request
const DefaultMaxTokens int64 = 4096

// AnthropicAPIVersion is the version header required for Anthropic API
const AnthropicAPIVersion = "2023-06-01"

// AnthropicMessagesPath is the path for Anthropic messages API
const AnthropicMessagesPath = "/v1/messages"

// HeaderKeyValue represents a header key-value pair
type HeaderKeyValue struct {
	Key   string
	Value string
}

// BuildRequestHeaders returns the headers needed for an Anthropic API request.
// messagesPath should include any base_url prefix (e.g. /Anthropic/v1/messages for AMD).
// When empty, AnthropicMessagesPath (/v1/messages) is used.
func BuildRequestHeaders(apiKey string, bodyLength int, messagesPath string) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, false)
}

func buildRequestHeaders(apiKey string, bodyLength int, messagesPath string, streaming bool) []HeaderKeyValue {
	if strings.TrimSpace(messagesPath) == "" {
		messagesPath = AnthropicMessagesPath
	}
	headers := []HeaderKeyValue{
		{Key: "anthropic-version", Value: AnthropicAPIVersion},
		{Key: "content-type", Value: "application/json"},
		{Key: "accept-encoding", Value: "identity"},
		{Key: ":path", Value: messagesPath},
		{Key: "content-length", Value: fmt.Sprintf("%d", bodyLength)},
	}
	if streaming {
		headers = append(headers, HeaderKeyValue{Key: "accept", Value: "text/event-stream"})
	}
	if strings.TrimSpace(apiKey) != "" {
		headers = append([]HeaderKeyValue{{Key: "x-api-key", Value: apiKey}}, headers...)
	}
	return headers
}

// HeadersToRemove returns headers that should be removed when routing to Anthropic.
func HeadersToRemove() []string {
	return []string{"authorization", "content-length"}
}

// ToAnthropicRequestBody transforms an OpenAI-format request to Anthropic API format (JSON).
// This is used for Envoy-routed requests where the router transforms the body
// before forwarding to Anthropic via Envoy. Equivalent to calling
// ToAnthropicRequestBodyWithPassthrough with a nil passthrough.
func ToAnthropicRequestBody(openAIRequest *openai.ChatCompletionNewParams) ([]byte, error) {
	return ToAnthropicRequestBodyWithPassthrough(openAIRequest, nil)
}

// ToAnthropicRequestBodyWithPassthrough transforms an OpenAI-format request to
// Anthropic API format and replays Anthropic-only fields carried in pt.
//
// Passing a nil passthrough is byte-identical to ToAnthropicRequestBody —
// existing callers and behaviour are preserved. When pt is non-nil, fields
// without an OpenAI representation (cache_control markers, top_k,
// metadata.user_id, multi-block system prompts, image blocks, tool_result
// error/array content) are emitted on the outbound body.
func ToAnthropicRequestBodyWithPassthrough(openAIRequest *openai.ChatCompletionNewParams, pt *AnthropicPassthrough) ([]byte, error) {
	systemPrompt, messages, err := buildAnthropicMessages(openAIRequest.Messages)
	if err != nil {
		return nil, err
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(openAIRequest.Model),
		Messages:  messages,
		MaxTokens: resolveMaxTokens(openAIRequest),
		System:    buildSystem(systemPrompt, pt),
	}

	if err := applyOpenAIToolsToAnthropicParams(&params, openAIRequest); err != nil {
		return nil, err
	}
	applyToolsCacheControl(&params, pt)
	applyMessagesCacheControl(params.Messages, pt)
	applyUserMessageImageBlocks(params.Messages, pt)
	applySampling(&params, openAIRequest, pt)
	applyMetadata(&params, openAIRequest, pt)

	return json.Marshal(params)
}

// resolveMaxTokens picks max_tokens with Anthropic's required default.
func resolveMaxTokens(req *openai.ChatCompletionNewParams) int64 {
	switch {
	case req.MaxCompletionTokens.Value > 0:
		return req.MaxCompletionTokens.Value
	case req.MaxTokens.Value > 0:
		return req.MaxTokens.Value
	default:
		return DefaultMaxTokens
	}
}

// buildSystem returns the Anthropic system array. When pt carries
// SystemBlocks (array-form system from inbound), each block is emitted with
// its cache_control marker; otherwise it falls back to the string-form system
// derived from the OpenAI messages, preserving today's behaviour byte-for-byte.
func buildSystem(systemPrompt string, pt *AnthropicPassthrough) []anthropic.TextBlockParam {
	if pt != nil && len(pt.SystemBlocks) > 0 {
		out := make([]anthropic.TextBlockParam, 0, len(pt.SystemBlocks))
		for _, sb := range pt.SystemBlocks {
			block := anthropic.TextBlockParam{Text: sb.Text}
			if sb.CacheControl != nil {
				block.CacheControl = toSDKCacheControl(*sb.CacheControl)
			}
			out = append(out, block)
		}
		return out
	}
	if systemPrompt == "" {
		return nil
	}
	return []anthropic.TextBlockParam{{Text: systemPrompt}}
}

// applySampling sets temperature, top_p, and stop_sequences from the OpenAI
// request, plus top_k from the passthrough when present.
func applySampling(params *anthropic.MessageNewParams, req *openai.ChatCompletionNewParams, pt *AnthropicPassthrough) {
	if req.Temperature.Valid() {
		params.Temperature = anthropic.Float(req.Temperature.Value)
	}
	if req.TopP.Valid() {
		params.TopP = anthropic.Float(req.TopP.Value)
	}
	if len(req.Stop.OfStringArray) > 0 {
		params.StopSequences = req.Stop.OfStringArray
	} else if req.Stop.OfString.Value != "" {
		params.StopSequences = []string{req.Stop.OfString.Value}
	}
	if pt != nil && pt.TopK != nil {
		params.TopK = anthropic.Int(*pt.TopK)
	}
}

// applyMetadata emits metadata.user_id from the passthrough when set, falling
// back to the OpenAI request's `user` field as the conventional mapping.
func applyMetadata(params *anthropic.MessageNewParams, req *openai.ChatCompletionNewParams, pt *AnthropicPassthrough) {
	userID := ""
	if pt != nil && pt.MetadataUserID != "" {
		userID = pt.MetadataUserID
	} else if req != nil && req.User.Valid() && req.User.Value != "" {
		userID = req.User.Value
	}
	if userID == "" {
		return
	}
	params.Metadata = anthropic.MetadataParam{UserID: anthropic.String(userID)}
}

// applyToolsCacheControl attaches cache_control markers to tool definitions
// based on the `tools[i]` keys in the passthrough.
func applyToolsCacheControl(params *anthropic.MessageNewParams, pt *AnthropicPassthrough) {
	if pt == nil || len(pt.CacheControl) == 0 {
		return
	}
	for i := range params.Tools {
		spec, ok := pt.CacheControl[fmt.Sprintf("tools[%d]", i)]
		if !ok {
			continue
		}
		if params.Tools[i].OfTool != nil {
			params.Tools[i].OfTool.CacheControl = toSDKCacheControl(spec)
		}
	}
}

// applyMessagesCacheControl attaches per-content-block cache_control markers
// to the outbound user/assistant messages. Markers whose `messages[i].content[j]`
// key no longer matches a block on the outbound side are silently dropped to
// match the surrounding lossy-translation discipline.
func applyMessagesCacheControl(messages []anthropic.MessageParam, pt *AnthropicPassthrough) {
	if pt == nil || len(pt.CacheControl) == 0 {
		return
	}
	for i := range messages {
		for j := range messages[i].Content {
			spec, ok := pt.CacheControl[fmt.Sprintf("messages[%d].content[%d]", i, j)]
			if !ok {
				continue
			}
			sdkCC := toSDKCacheControl(spec)
			block := &messages[i].Content[j]
			switch {
			case block.OfText != nil:
				block.OfText.CacheControl = sdkCC
			case block.OfImage != nil:
				block.OfImage.CacheControl = sdkCC
			case block.OfToolUse != nil:
				block.OfToolUse.CacheControl = sdkCC
			case block.OfToolResult != nil:
				block.OfToolResult.CacheControl = sdkCC
			}
		}
	}
}

// toSDKCacheControl converts the package-local CacheControlSpec into the SDK's
// CacheControlEphemeralParam. Type defaults to "ephemeral" (the only value the
// SDK supports today); TTL is forwarded when set.
func toSDKCacheControl(spec CacheControlSpec) anthropic.CacheControlEphemeralParam {
	cc := anthropic.NewCacheControlEphemeralParam()
	if spec.TTL != "" {
		cc.TTL = anthropic.CacheControlEphemeralTTL(spec.TTL)
	}
	return cc
}

// ToOpenAIResponseBody transforms an Anthropic API response to OpenAI format.
// This is used for Envoy-routed requests where the router transforms the response
// after receiving it from Anthropic via Envoy.
func ToOpenAIResponseBody(anthropicResponse []byte, model string) ([]byte, error) {
	logging.Debugf("Raw Anthropic response (%d bytes): %s", len(anthropicResponse), string(anthropicResponse))

	var resp anthropic.Message
	if err := json.Unmarshal(anthropicResponse, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	logging.Debugf("Parsed Anthropic response - ID: %s, Content blocks: %d, StopReason: %s", resp.ID, len(resp.Content), resp.StopReason)

	toolCalls, content := openAIToolCallsFromAnthropicContent(resp.Content)

	// Map stop reason
	finishReason := "stop"
	switch resp.StopReason {
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	}
	if len(toolCalls) > 0 && finishReason == "stop" {
		finishReason = "tool_calls"
	}

	openAIResp := &openai.ChatCompletion{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:      "assistant",
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}

	return json.Marshal(openAIResp)
}

// extractSystemContent extracts text from a system message
func extractSystemContent(msg *openai.ChatCompletionSystemMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}

// extractUserContent extracts text from a user message. Kept for
// backwards-compatibility with callers that only need the flattened text
// representation; userMessageBlocks is the structured form used by the
// outbound writer (which also preserves image blocks).
func extractUserContent(msg *openai.ChatCompletionUserMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

// userMessageBlocks converts an OpenAI user message into a slice of Anthropic
// content blocks, preserving image_url parts as Anthropic image blocks. This
// fixes the long-standing drop where image content (OpenAI-shape OfImageURL)
// was flattened to an empty string by extractUserContent, causing the model to
// report "I don't see any image attached" on the upstream side.
func userMessageBlocks(msg *openai.ChatCompletionUserMessageParam) []anthropic.ContentBlockParamUnion {
	if msg.Content.OfString.Value != "" {
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(msg.Content.OfString.Value)}
	}
	var blocks []anthropic.ContentBlockParamUnion
	var textParts []string
	flushText := func() {
		if len(textParts) > 0 {
			blocks = append(blocks, anthropic.NewTextBlock(strings.Join(textParts, " ")))
			textParts = nil
		}
	}
	for _, part := range msg.Content.OfArrayOfContentParts {
		switch {
		case part.OfText != nil:
			textParts = append(textParts, part.OfText.Text)
		case part.OfImageURL != nil:
			flushText()
			if block, ok := openAIImageURLToAnthropicBlock(part.OfImageURL.ImageURL.URL); ok {
				blocks = append(blocks, block)
			}
		}
	}
	flushText()
	return blocks
}

// openAIImageURLToAnthropicBlock parses an OpenAI image_url value into an
// Anthropic image block. Both data-URI (data:image/<type>;base64,<data>) and
// plain URL forms are supported; unrecognised shapes are skipped.
func openAIImageURLToAnthropicBlock(url string) (anthropic.ContentBlockParamUnion, bool) {
	const dataURIPrefix = "data:"
	if strings.HasPrefix(url, dataURIPrefix) {
		mediaType, data, ok := parseDataURI(url)
		if !ok {
			return anthropic.ContentBlockParamUnion{}, false
		}
		return anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
			Data:      data,
			MediaType: anthropic.Base64ImageSourceMediaType(mediaType),
		}), true
	}
	if url == "" {
		return anthropic.ContentBlockParamUnion{}, false
	}
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: url}), true
}

// parseDataURI extracts the media type and base64-encoded payload from a data
// URI of the form `data:<media-type>;base64,<data>`. Returns ok=false for any
// other shape.
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	const prefix = "data:"
	if !strings.HasPrefix(uri, prefix) {
		return "", "", false
	}
	rest := uri[len(prefix):]
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", "", false
	}
	header := rest[:comma]
	payload := rest[comma+1:]
	if !strings.Contains(header, ";base64") {
		return "", "", false
	}
	mediaType = strings.TrimSuffix(header, ";base64")
	if mediaType == "" {
		return "", "", false
	}
	return mediaType, payload, true
}

// applyUserMessageImageBlocks appends image blocks captured in the passthrough
// to the matching user message. Indexing is user-message-only (index 0 = first
// user message in messages[]). When the index is out of range, images are
// silently dropped to match the surrounding lossy-translation discipline.
func applyUserMessageImageBlocks(messages []anthropic.MessageParam, pt *AnthropicPassthrough) {
	if pt == nil || len(pt.UserMessageImageBlocks) == 0 {
		return
	}
	userIdx := -1
	for i := range messages {
		if messages[i].Role != anthropic.MessageParamRoleUser {
			continue
		}
		userIdx++
		images, ok := pt.UserMessageImageBlocks[userIdx]
		if !ok {
			continue
		}
		for _, img := range images {
			block, ok := buildImageBlockFromSource(img.Source)
			if !ok {
				continue
			}
			messages[i].Content = append(messages[i].Content, block)
		}
	}
}

func buildImageBlockFromSource(src ImageSource) (anthropic.ContentBlockParamUnion, bool) {
	switch src.Type {
	case "base64":
		if src.Data == "" || src.MediaType == "" {
			return anthropic.ContentBlockParamUnion{}, false
		}
		return anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
			Data:      src.Data,
			MediaType: anthropic.Base64ImageSourceMediaType(src.MediaType),
		}), true
	case "url":
		if src.URL == "" {
			return anthropic.ContentBlockParamUnion{}, false
		}
		return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: src.URL}), true
	}
	return anthropic.ContentBlockParamUnion{}, false
}

// extractAssistantContent extracts text from an assistant message
func extractAssistantContent(msg *openai.ChatCompletionAssistantMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

func buildAnthropicMessages(
	messages []openai.ChatCompletionMessageParamUnion,
) (string, []anthropic.MessageParam, error) {
	var systemPrompt string
	var anthropicMessages []anthropic.MessageParam

	for _, msg := range messages {
		switch {
		case msg.OfSystem != nil:
			systemPrompt = extractSystemContent(msg.OfSystem)
		case msg.OfUser != nil:
			blocks := userMessageBlocks(msg.OfUser)
			if len(blocks) > 0 {
				anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(blocks...))
			}
		case msg.OfAssistant != nil:
			blocks, err := assistantContentBlocks(msg.OfAssistant)
			if err != nil {
				return "", nil, err
			}
			if len(blocks) > 0 {
				anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(blocks...))
			}
		case msg.OfTool != nil:
			toolUseID := strings.TrimSpace(msg.OfTool.ToolCallID)
			if toolUseID == "" {
				return "", nil, fmt.Errorf("tool message missing tool_call_id")
			}
			content := extractToolMessageContent(msg.OfTool)
			anthropicMessages = append(anthropicMessages, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(toolUseID, content, false),
			))
		default:
			logging.Debugf("Skipping unsupported OpenAI message role in Anthropic conversion")
		}
	}

	return systemPrompt, anthropicMessages, nil
}

func extractToolMessageContent(msg *openai.ChatCompletionToolMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}

func assistantContentBlocks(
	msg *openai.ChatCompletionAssistantMessageParam,
) ([]anthropic.ContentBlockParamUnion, error) {
	var blocks []anthropic.ContentBlockParamUnion

	content := extractAssistantContent(msg)
	if content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(content))
	}

	for _, toolCall := range msg.ToolCalls {
		toolUseID := strings.TrimSpace(toolCall.ID)
		if toolUseID == "" {
			return nil, fmt.Errorf("assistant tool_call missing id")
		}
		name := strings.TrimSpace(toolCall.Function.Name)
		if name == "" {
			return nil, fmt.Errorf("assistant tool_call %s missing function name", toolUseID)
		}
		input, err := parseToolCallArguments(toolCall.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("assistant tool_call %s: %w", toolUseID, err)
		}
		blocks = append(blocks, anthropic.NewToolUseBlock(toolUseID, input, name))
	}

	return blocks, nil
}

func parseToolCallArguments(arguments string) (any, error) {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return map[string]any{}, nil
	}
	var parsed any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return nil, fmt.Errorf("invalid tool arguments JSON: %w", err)
	}
	return parsed, nil
}

func applyOpenAIToolsToAnthropicParams(
	params *anthropic.MessageNewParams,
	openAIRequest *openai.ChatCompletionNewParams,
) error {
	if openAIRequest == nil {
		return nil
	}

	if len(openAIRequest.Tools) > 0 {
		tools, err := openAIToolsToAnthropic(openAIRequest.Tools)
		if err != nil {
			return err
		}
		params.Tools = tools
	}

	toolChoice, hasChoice := openAIToolChoiceToAnthropic(openAIRequest.ToolChoice)
	if hasChoice {
		params.ToolChoice = toolChoice
	}
	return nil
}

func openAIToolsToAnthropic(tools []openai.ChatCompletionToolParam) ([]anthropic.ToolUnionParam, error) {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for index, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			logging.Debugf("Skipping non-function OpenAI tool type %q at index %d", tool.Type, index)
			continue
		}
		name := strings.TrimSpace(tool.Function.Name)
		if name == "" {
			return nil, fmt.Errorf("tools[%d]: function name is required", index)
		}

		inputSchema, err := functionParametersToAnthropicSchema(tool.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("tools[%d] (%s): %w", index, name, err)
		}

		toolParam := anthropic.ToolParam{
			Name:        name,
			InputSchema: inputSchema,
			Type:        anthropic.ToolTypeCustom,
		}
		if desc := strings.TrimSpace(tool.Function.Description.Value); desc != "" {
			toolParam.Description = anthropic.String(desc)
		}
		result = append(result, anthropic.ToolUnionParam{OfTool: &toolParam})
	}
	return result, nil
}

func functionParametersToAnthropicSchema(
	parameters openai.FunctionParameters,
) (anthropic.ToolInputSchemaParam, error) {
	schema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}
	if parameters == nil {
		return schema, nil
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		return schema, fmt.Errorf("marshal function parameters: %w", err)
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return schema, fmt.Errorf("parse function parameters: %w", err)
	}
	if schema.Type == "" {
		schema.Type = "object"
	}
	return schema, nil
}

func openAIToolChoiceToAnthropic(
	choice openai.ChatCompletionToolChoiceOptionUnionParam,
) (anthropic.ToolChoiceUnionParam, bool) {
	if !param.IsOmitted(choice.OfAuto) {
		value := strings.TrimSpace(choice.OfAuto.Value)
		switch value {
		case "", "auto":
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{},
			}, true
		case "none":
			none := anthropic.NewToolChoiceNoneParam()
			return anthropic.ToolChoiceUnionParam{OfNone: &none}, true
		case "required", "any":
			return anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{},
			}, true
		default:
			logging.Debugf("Unknown OpenAI tool_choice auto value %q, defaulting to auto", value)
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{},
			}, true
		}
	}

	if choice.OfChatCompletionNamedToolChoice != nil {
		name := strings.TrimSpace(choice.OfChatCompletionNamedToolChoice.Function.Name)
		if name == "" {
			return anthropic.ToolChoiceUnionParam{}, false
		}
		return anthropic.ToolChoiceParamOfTool(name), true
	}

	return anthropic.ToolChoiceUnionParam{}, false
}

func openAIToolCallsFromAnthropicContent(
	blocks []anthropic.ContentBlockUnion,
) ([]openai.ChatCompletionMessageToolCall, string) {
	var contentParts []string
	var toolCalls []openai.ChatCompletionMessageToolCall

	for _, block := range blocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				contentParts = append(contentParts, block.Text)
			}
		case "tool_use":
			arguments := string(block.Input)
			if len(block.Input) == 0 {
				arguments = "{}"
			}
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      block.Name,
					Arguments: arguments,
				},
			})
		default:
			logging.Debugf("Skipping unsupported Anthropic content block type %q in OpenAI response conversion", block.Type)
		}
	}

	return toolCalls, strings.Join(contentParts, "")
}
