// Package-level request-body construction helpers for the Anthropic format.
//
// This file contains the private helper functions that implement the
// OpenAI→Anthropic request body translation called by the public entry
// points ToAnthropicRequestBody and ToAnthropicRequestBodyWithPassthrough
// (in client.go). Keeping them here lets client.go stay focused on the
// public API surface and the symmetric response-path helpers.
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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

// applyToolResultPassthrough overwrites tool_result blocks with the passthrough
// payload: is_error: true is restored from pt.ToolResultErrors, and array
// content (text + image blocks) is restored from pt.ToolResultArrayContent.
// Matches by tool_use_id; missing keys leave the block as today's
// translation produced it.
func applyToolResultPassthrough(messages []anthropic.MessageParam, pt *AnthropicPassthrough) {
	if pt == nil || (len(pt.ToolResultErrors) == 0 && len(pt.ToolResultArrayContent) == 0) {
		return
	}
	for i := range messages {
		for j := range messages[i].Content {
			tr := messages[i].Content[j].OfToolResult
			if tr == nil || tr.ToolUseID == "" {
				continue
			}
			if isErr, ok := pt.ToolResultErrors[tr.ToolUseID]; ok && isErr {
				tr.IsError = anthropic.Bool(true)
			}
			if blocks, ok := pt.ToolResultArrayContent[tr.ToolUseID]; ok && len(blocks) > 0 {
				tr.Content = toSDKToolResultContent(blocks)
			}
		}
	}
}

func toSDKToolResultContent(blocks []ToolResultContentBlock) []anthropic.ToolResultBlockParamContentUnion {
	out := make([]anthropic.ToolResultBlockParamContentUnion, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case "text":
			out = append(out, anthropic.ToolResultBlockParamContentUnion{
				OfText: &anthropic.TextBlockParam{Text: b.Text},
			})
		case "image":
			if b.Source == nil {
				continue
			}
			img, ok := buildImageParamFromSource(*b.Source)
			if !ok {
				continue
			}
			out = append(out, anthropic.ToolResultBlockParamContentUnion{OfImage: img})
		}
	}
	return out
}

// buildImageParamFromSource is the variant of buildImageBlockFromSource that
// returns the raw *ImageBlockParam needed by ToolResultBlockParamContentUnion
// (which embeds ImageBlockParam directly rather than via NewImageBlock).
func buildImageParamFromSource(src ImageSource) (*anthropic.ImageBlockParam, bool) {
	switch src.Type {
	case "base64":
		if src.Data == "" || src.MediaType == "" {
			return nil, false
		}
		return &anthropic.ImageBlockParam{
			Source: anthropic.ImageBlockParamSourceUnion{
				OfBase64: &anthropic.Base64ImageSourceParam{
					Data:      src.Data,
					MediaType: anthropic.Base64ImageSourceMediaType(src.MediaType),
				},
			},
		}, true
	case "url":
		if src.URL == "" {
			return nil, false
		}
		return &anthropic.ImageBlockParam{
			Source: anthropic.ImageBlockParamSourceUnion{
				OfURL: &anthropic.URLImageSourceParam{URL: src.URL},
			},
		}, true
	}
	return nil, false
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
