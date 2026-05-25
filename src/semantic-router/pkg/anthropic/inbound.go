package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"github.com/openai/openai-go/shared/constant"
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// SourceProtocolAnthropic is the SourceProtocol stamp set on IRExtensions
// by ParseAnthropicRequest. Mirrors config.ClientProtocolAnthropic but is
// duplicated here to avoid creating an extproc → config → anthropic cycle.
const SourceProtocolAnthropic = "anthropic"

// ParseAnthropicRequest translates a raw Anthropic /v1/messages JSON body
// into the protocol-neutral IR pair (*openai.ChatCompletionNewParams plus
// *ir.IRExtensions).
//
// Behavior contract:
//
//   - Returns a non-nil error only when the JSON is structurally invalid
//     or a required field is missing.
//   - Unknown block types (e.g. redacted_thinking, search_result) are
//     warn-and-dropped via IRExtensions.Warnings; they never fail the parse.
//   - cache_control markers on any block are captured into
//     IRExtensions.CacheControl AND the marked block continues to be
//     translated into the OpenAI IR (the marker is a sidecar, not a
//     gate).
//   - tool_result.is_error is preserved as an IRExtensions warning
//     (Reason="tool_result_is_error", Detail=tool_use_id) so the
//     round-trip back through ToAnthropicRequestBody can restore the
//     flag.
//   - Array-form tool_result content (text + image parts) is preserved as
//     OpenAI tool-message array content; only the text payload survives
//     today because the OpenAI tool-message shape does not carry image
//     parts, but the image content is recorded into Warnings so an
//     Anthropic-backend emitter can reconstruct it.
//   - The OpenAI request always has a populated MaxTokens (Anthropic
//     requires it; we default to DefaultMaxTokens=4096 when absent to
//     match the symmetric outbound emitter).
func ParseAnthropicRequest(body []byte) (*openai.ChatCompletionNewParams, *ir.IRExtensions, error) {
	if !gjson.ValidBytes(body) {
		return nil, nil, fmt.Errorf("anthropic request body is not valid JSON")
	}

	model := gjson.GetBytes(body, "model")
	if !model.Exists() || model.Type != gjson.String || model.String() == "" {
		return nil, nil, fmt.Errorf("anthropic request: model field is required")
	}

	params := &openai.ChatCompletionNewParams{
		Model: model.String(),
	}
	ext := &ir.IRExtensions{SourceProtocol: SourceProtocolAnthropic}

	convertSamplingParams(body, params, ext)
	convertSystem(gjson.GetBytes(body, "system"), params, ext)
	convertMessages(gjson.GetBytes(body, "messages"), params, ext)
	convertTools(gjson.GetBytes(body, "tools"), params, ext)
	convertToolChoice(gjson.GetBytes(body, "tool_choice"), params, ext)
	convertMetadata(gjson.GetBytes(body, "metadata"), params, ext)
	convertThinking(gjson.GetBytes(body, "thinking"), ext)
	captureTopLevelCacheControl(body, ext)

	return params, ext, nil
}

func convertSamplingParams(body []byte, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	params.MaxTokens = openai.Int(readMaxTokens(body))
	applyNumericSampling(body, params, ext)
	applyStopSequences(body, params)
	if beta := gjson.GetBytes(body, "anthropic_beta"); beta.Exists() && beta.Type == gjson.String {
		ext.AnthropicBeta = beta.String()
	}
}

func readMaxTokens(body []byte) int64 {
	if mt := gjson.GetBytes(body, "max_tokens"); mt.Exists() && mt.Type == gjson.Number {
		return mt.Int()
	}
	return DefaultMaxTokens
}

func applyNumericSampling(body []byte, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if t := gjson.GetBytes(body, "temperature"); t.Exists() && t.Type == gjson.Number {
		params.Temperature = openai.Float(t.Float())
	}
	if tp := gjson.GetBytes(body, "top_p"); tp.Exists() && tp.Type == gjson.Number {
		params.TopP = openai.Float(tp.Float())
	}
	if tk := gjson.GetBytes(body, "top_k"); tk.Exists() && tk.Type == gjson.Number {
		v := tk.Int()
		ext.TopK = &v
	}
}

func applyStopSequences(body []byte, params *openai.ChatCompletionNewParams) {
	stop := gjson.GetBytes(body, "stop_sequences")
	if !stop.Exists() || !stop.IsArray() {
		return
	}
	var seqs []string
	stop.ForEach(func(_, v gjson.Result) bool {
		if v.Type == gjson.String && v.String() != "" {
			seqs = append(seqs, v.String())
		}
		return true
	})
	if len(seqs) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: seqs}
	}
}

func convertSystem(system gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if !system.Exists() {
		return
	}

	if system.Type == gjson.String {
		text := system.String()
		if text != "" {
			params.Messages = append(params.Messages, openai.SystemMessage(text))
		}
		return
	}
	if !system.IsArray() {
		ext.AppendWarning(ir.Warning{
			Field:    "system",
			Reason:   "unsupported_type",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("system must be string or array, got %s", system.Type.String()),
		})
		return
	}

	var parts []openai.ChatCompletionContentPartTextParam
	system.ForEach(func(idx, block gjson.Result) bool {
		blockID := fmt.Sprintf("system.%d", idx.Int())
		text := block.Get("text").String()
		if text == "" {
			return true
		}
		parts = append(parts, openai.ChatCompletionContentPartTextParam{Text: text})

		spec := SystemBlockSpec(blockID, text, block)
		ext.SystemBlocks = append(ext.SystemBlocks, spec)
		captureCacheControl(blockID, block.Get("cache_control"), ext)
		return true
	})
	if len(parts) > 0 {
		params.Messages = append(params.Messages, openai.SystemMessage(parts))
	}
}

// SystemBlockSpec is exported so emitters in the same package can rebuild
// outbound system arrays without duplicating the field-mapping logic.
func SystemBlockSpec(blockID, text string, block gjson.Result) ir.SystemBlock {
	var cc *ir.CacheControlSpec
	if raw := block.Get("cache_control"); raw.Exists() {
		spec := parseCacheControlSpec(raw)
		cc = &spec
	}
	return ir.SystemBlock{BlockID: blockID, Text: text, CacheControl: cc}
}

func convertMessages(messages gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if !messages.Exists() || !messages.IsArray() {
		return
	}
	messages.ForEach(func(idx, msg gjson.Result) bool {
		convertMessage(int(idx.Int()), msg, params, ext)
		return true
	})
}

func convertMessage(msgIdx int, msg gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	role := msg.Get("role").String()
	content := msg.Get("content")

	switch role {
	case "user":
		convertUserMessage(msgIdx, content, params, ext)
	case "assistant":
		convertAssistantMessage(msgIdx, content, params, ext)
	default:
		ext.AppendWarning(ir.Warning{
			Field:    fmt.Sprintf("messages[%d].role", msgIdx),
			Reason:   "unsupported_role",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("role %q is not user or assistant", role),
		})
	}
}

func convertUserMessage(msgIdx int, content gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	// Anthropic user-turn content is either a plain string, in which case
	// it becomes one OpenAI user message, or an array of blocks. The array
	// case is heterogeneous: text/image parts contribute to the user
	// message, while tool_result blocks emit their own tool-role messages
	// in document order alongside (or replacing) the user message.
	if content.Type == gjson.String {
		text := content.String()
		if text != "" {
			params.Messages = append(params.Messages, openai.UserMessage(text))
		}
		return
	}
	if !content.IsArray() {
		return
	}

	var userParts []openai.ChatCompletionContentPartUnionParam

	flushUserParts := func() {
		if len(userParts) == 0 {
			return
		}
		params.Messages = append(params.Messages, openai.UserMessage(userParts))
		userParts = nil
	}

	content.ForEach(func(idx, block gjson.Result) bool {
		blockID := fmt.Sprintf("messages[%d].content[%d]", msgIdx, idx.Int())
		captureCacheControl(blockID, block.Get("cache_control"), ext)
		userParts = dispatchUserBlock(blockID, block, userParts, params, ext, flushUserParts)
		return true
	})

	flushUserParts()
}

// dispatchAssistantBlock applies one assistant-turn content block.
// text → textParts (one OfText entry per block, preserving order),
// tool_use → toolCalls (assembled into ToolCalls on the assistant message),
// thinking → ext.ThinkingSignatures for round-trip,
// redacted_thinking / server_tool_use / unknown → ext.Warnings.
func dispatchAssistantBlock(
	blockID string,
	block gjson.Result,
	textParts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion,
	toolCalls []openai.ChatCompletionMessageToolCallParam,
	ext *ir.IRExtensions,
) (
	[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion,
	[]openai.ChatCompletionMessageToolCallParam,
) {
	switch block.Get("type").String() {
	case "text":
		if text := block.Get("text").String(); text != "" {
			textParts = append(textParts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: text},
			})
		}
	case "tool_use":
		if call, ok := convertToolUseBlock(blockID, block, ext); ok {
			toolCalls = append(toolCalls, call)
		}
	case "thinking":
		captureThinkingBlock(blockID, block, ext)
	case "redacted_thinking":
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".type",
			Reason:   "unsupported_block_type",
			Severity: ir.WarningSeverityLossy,
			Detail:   "redacted_thinking blocks are dropped (out of scope for PR2)",
		})
	case "server_tool_use":
		captureServerToolUse(blockID, block, ext)
	default:
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".type",
			Reason:   "unsupported_block_type",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("assistant-turn block type %q is not handled", block.Get("type").String()),
		})
	}
	return textParts, toolCalls
}

// dispatchUserBlock applies one user-turn content block. Inline parts
// (text/image) accumulate into userParts; tool_result blocks flush and
// emit a separate tool-role message in document order; sidecar-only
// blocks (document/thinking/server_tool_use) are captured into ext.
func dispatchUserBlock(
	blockID string,
	block gjson.Result,
	userParts []openai.ChatCompletionContentPartUnionParam,
	params *openai.ChatCompletionNewParams,
	ext *ir.IRExtensions,
	flushUserParts func(),
) []openai.ChatCompletionContentPartUnionParam {
	switch block.Get("type").String() {
	case "text":
		if text := block.Get("text").String(); text != "" {
			userParts = append(userParts, openai.TextContentPart(text))
		}
	case "image":
		if part, ok := convertImageBlock(blockID, block, ext); ok {
			userParts = append(userParts, part)
		}
	case "tool_result":
		// tool_result terminates the running user-message batch and emits
		// a tool-role message in place so the original document order
		// user[text] → tool_result → user[text] becomes the equivalent
		// sequence of OpenAI messages.
		flushUserParts()
		if toolMsg, ok := convertToolResultBlock(blockID, block, ext); ok {
			params.Messages = append(params.Messages, toolMsg)
		}
	case "document":
		captureDocumentBlock(blockID, block, ext)
	case "thinking", "redacted_thinking":
		captureThinkingBlock(blockID, block, ext)
	case "server_tool_use":
		captureServerToolUse(blockID, block, ext)
	default:
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".type",
			Reason:   "unsupported_block_type",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("user-turn block type %q is not handled", block.Get("type").String()),
		})
	}
	return userParts
}

func convertAssistantMessage(msgIdx int, content gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	asst := openai.ChatCompletionAssistantMessageParam{Role: constant.Assistant("assistant")}

	if content.Type == gjson.String {
		text := content.String()
		if text != "" {
			asst.Content.OfString = param.NewOpt(text)
		}
		params.Messages = append(params.Messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &asst})
		return
	}
	if !content.IsArray() {
		params.Messages = append(params.Messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &asst})
		return
	}

	var textParts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion
	var toolCalls []openai.ChatCompletionMessageToolCallParam

	content.ForEach(func(idx, block gjson.Result) bool {
		blockID := fmt.Sprintf("messages[%d].content[%d]", msgIdx, idx.Int())
		captureCacheControl(blockID, block.Get("cache_control"), ext)
		textParts, toolCalls = dispatchAssistantBlock(blockID, block, textParts, toolCalls, ext)
		return true
	})

	if len(textParts) > 0 {
		asst.Content.OfArrayOfContentParts = textParts
	}
	if len(toolCalls) > 0 {
		asst.ToolCalls = toolCalls
	}

	params.Messages = append(params.Messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &asst})
}

func convertImageBlock(blockID string, block gjson.Result, ext *ir.IRExtensions) (openai.ChatCompletionContentPartUnionParam, bool) {
	source := block.Get("source")
	if !source.Exists() {
		ext.AppendWarning(ir.Warning{Field: blockID + ".source", Reason: "missing_source", Severity: ir.WarningSeverityLossy})
		return openai.ChatCompletionContentPartUnionParam{}, false
	}
	switch source.Get("type").String() {
	case "base64":
		mediaType := source.Get("media_type").String()
		data := source.Get("data").String()
		if mediaType == "" || data == "" {
			ext.AppendWarning(ir.Warning{Field: blockID + ".source", Reason: "incomplete_base64", Severity: ir.WarningSeverityLossy})
			return openai.ChatCompletionContentPartUnionParam{}, false
		}
		url := fmt.Sprintf("data:%s;base64,%s", mediaType, data)
		return openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: url}), true
	case "url":
		url := source.Get("url").String()
		if url == "" {
			ext.AppendWarning(ir.Warning{Field: blockID + ".source.url", Reason: "missing_url", Severity: ir.WarningSeverityLossy})
			return openai.ChatCompletionContentPartUnionParam{}, false
		}
		return openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: url}), true
	case "file", "file_id":
		fileID := source.Get("file_id").String()
		if fileID == "" {
			fileID = source.Get("id").String()
		}
		if fileID == "" {
			ext.AppendWarning(ir.Warning{Field: blockID + ".source.file_id", Reason: "missing_file_id", Severity: ir.WarningSeverityLossy})
			return openai.ChatCompletionContentPartUnionParam{}, false
		}
		// OpenAI image_url does not have a file_id concept; we synthesize
		// a file_id: URI scheme so the IR remains self-describing and warn
		// so any OpenAI-backend emitter knows to resolve or drop it.
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".source",
			Reason:   "image_file_id_unresolved",
			Severity: ir.WarningSeverityInfo,
			Detail:   "image source type=file_id requires backend-specific resolution",
		})
		return openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: "file_id:" + fileID}), true
	default:
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".source.type",
			Reason:   "unsupported_image_source",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("image source.type %q is not handled", source.Get("type").String()),
		})
		return openai.ChatCompletionContentPartUnionParam{}, false
	}
}

func convertToolResultBlock(blockID string, block gjson.Result, ext *ir.IRExtensions) (openai.ChatCompletionMessageParamUnion, bool) {
	toolUseID := strings.TrimSpace(block.Get("tool_use_id").String())
	if toolUseID == "" {
		ext.AppendWarning(ir.Warning{Field: blockID + ".tool_use_id", Reason: "missing_tool_use_id", Severity: ir.WarningSeverityError})
		return openai.ChatCompletionMessageParamUnion{}, false
	}
	isError := block.Get("is_error").Bool()

	textContent, imageCount := flattenToolResultContent(blockID, block.Get("content"), ext)

	toolMsg := openai.ChatCompletionToolMessageParam{
		Role:       constant.Tool("tool"),
		ToolCallID: toolUseID,
	}
	if textContent != "" {
		toolMsg.Content = openai.ChatCompletionToolMessageParamContentUnion{
			OfString: param.NewOpt(textContent),
		}
	}

	if isError {
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".is_error",
			Reason:   "tool_result_is_error",
			Severity: ir.WarningSeverityInfo,
			Detail:   toolUseID,
		})
	}
	if imageCount > 0 {
		ext.AppendWarning(ir.Warning{
			Field:    blockID + ".content",
			Reason:   "tool_result_image_dropped",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("%d image part(s) on tool_result %s preserved only in IRExtensions", imageCount, toolUseID),
		})
	}

	return openai.ChatCompletionMessageParamUnion{OfTool: &toolMsg}, true
}

// flattenToolResultContent walks a tool_result content field and returns
// the concatenated text payload plus the count of image parts seen. Image
// parts are not surfaced into the OpenAI tool message (the shape does not
// support them) but the count drives a lossy warning so the outbound side
// can reconstruct or escalate.
func flattenToolResultContent(blockID string, content gjson.Result, ext *ir.IRExtensions) (string, int) {
	if !content.Exists() {
		return "", 0
	}
	if content.Type == gjson.String {
		return content.String(), 0
	}
	if !content.IsArray() {
		return "", 0
	}

	var texts []string
	imageCount := 0
	content.ForEach(func(idx, part gjson.Result) bool {
		partID := fmt.Sprintf("%s.content[%d]", blockID, idx.Int())
		switch part.Get("type").String() {
		case "text":
			if t := part.Get("text").String(); t != "" {
				texts = append(texts, t)
			}
		case "image":
			imageCount++
		case "document":
			ext.AppendWarning(ir.Warning{
				Field:    partID,
				Reason:   "tool_result_document_dropped",
				Severity: ir.WarningSeverityLossy,
			})
		default:
			ext.AppendWarning(ir.Warning{
				Field:    partID,
				Reason:   "unsupported_tool_result_part",
				Severity: ir.WarningSeverityLossy,
				Detail:   fmt.Sprintf("type %q", part.Get("type").String()),
			})
		}
		return true
	})
	if len(texts) == 1 {
		return texts[0], imageCount
	}
	return strings.Join(texts, "\n"), imageCount
}

func convertToolUseBlock(blockID string, block gjson.Result, ext *ir.IRExtensions) (openai.ChatCompletionMessageToolCallParam, bool) {
	id := strings.TrimSpace(block.Get("id").String())
	name := strings.TrimSpace(block.Get("name").String())
	if id == "" || name == "" {
		ext.AppendWarning(ir.Warning{
			Field:    blockID,
			Reason:   "incomplete_tool_use",
			Severity: ir.WarningSeverityError,
			Detail:   "tool_use requires id and name",
		})
		return openai.ChatCompletionMessageToolCallParam{}, false
	}

	args := "{}"
	if input := block.Get("input"); input.Exists() {
		// gjson Raw preserves the original JSON shape; if the SDK that
		// produced the body emitted a structured object, we keep it.
		args = input.Raw
	}

	return openai.ChatCompletionMessageToolCallParam{
		ID:   id,
		Type: constant.Function("function"),
		Function: openai.ChatCompletionMessageToolCallFunctionParam{
			Name:      name,
			Arguments: args,
		},
	}, true
}

func convertTools(tools gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if !tools.Exists() || !tools.IsArray() {
		return
	}
	var out []openai.ChatCompletionToolParam
	tools.ForEach(func(idx, tool gjson.Result) bool {
		path := fmt.Sprintf("tools[%d]", idx.Int())
		toolType := tool.Get("type").String()
		switch toolType {
		case "", "custom":
			if fp, ok := convertCustomTool(path, tool, ext); ok {
				out = append(out, fp)
			}
		default:
			captureServerToolDefinition(path, toolType, tool, ext)
		}
		return true
	})
	if len(out) > 0 {
		params.Tools = out
	}
}

func convertCustomTool(path string, tool gjson.Result, ext *ir.IRExtensions) (openai.ChatCompletionToolParam, bool) {
	name := strings.TrimSpace(tool.Get("name").String())
	if name == "" {
		ext.AppendWarning(ir.Warning{Field: path + ".name", Reason: "missing_name", Severity: ir.WarningSeverityError})
		return openai.ChatCompletionToolParam{}, false
	}
	fp := openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{Name: name},
	}
	if desc := tool.Get("description"); desc.Exists() && desc.String() != "" {
		fp.Function.Description = openai.String(desc.String())
	}
	if schema := tool.Get("input_schema"); schema.Exists() {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(schema.Raw), &parsed); err == nil {
			fp.Function.Parameters = parsed
		} else {
			ext.AppendWarning(ir.Warning{
				Field:    path + ".input_schema",
				Reason:   "invalid_schema",
				Severity: ir.WarningSeverityLossy,
				Detail:   err.Error(),
			})
		}
	}
	if strict := tool.Get("strict"); strict.Exists() && strict.Type == gjson.True {
		ext.SetToolStrict(name, true)
	}
	return fp, true
}

func captureServerToolDefinition(path, toolType string, tool gjson.Result, ext *ir.IRExtensions) {
	spec := ir.ServerToolSpec{
		Type: toolType,
		Name: tool.Get("name").String(),
	}
	if schema := tool.Get("input_schema"); schema.Exists() {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(schema.Raw), &parsed); err == nil {
			spec.Parameters = parsed
		}
	}
	ext.ServerTools = append(ext.ServerTools, spec)
	ext.AppendWarning(ir.Warning{
		Field:    path + ".type",
		Reason:   "server_tool_captured",
		Severity: ir.WarningSeverityInfo,
		Detail:   fmt.Sprintf("server-tool type %q preserved in IRExtensions", toolType),
	})
}

func convertToolChoice(tc gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if !tc.Exists() {
		return
	}
	tcType := tc.Get("type").String()
	switch tcType {
	case "auto":
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("auto"),
		}
	case "none":
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("none"),
		}
	case "any":
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("required"),
		}
	case "tool":
		name := tc.Get("name").String()
		if name != "" {
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfChatCompletionNamedToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
					Type: constant.Function("function"),
					Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: name,
					},
				},
			}
		}
	default:
		ext.AppendWarning(ir.Warning{
			Field:    "tool_choice.type",
			Reason:   "unsupported_tool_choice",
			Severity: ir.WarningSeverityLossy,
			Detail:   fmt.Sprintf("tool_choice.type %q is not handled", tcType),
		})
	}
	if disable := tc.Get("disable_parallel_tool_use"); disable.Exists() && disable.Type == gjson.True {
		ext.ToolChoiceDisableParallel = true
		// Mirror to the OpenAI native parallel_tool_calls field (the OAI
		// inverse of the Anthropic disable flag).
		params.ParallelToolCalls = openai.Bool(false)
	}
}

func convertMetadata(metadata gjson.Result, params *openai.ChatCompletionNewParams, ext *ir.IRExtensions) {
	if !metadata.Exists() {
		return
	}
	if uid := metadata.Get("user_id"); uid.Exists() && uid.Type == gjson.String && uid.String() != "" {
		ext.MetadataUserID = uid.String()
		params.User = openai.String(uid.String())
	}
}

func convertThinking(thinking gjson.Result, ext *ir.IRExtensions) {
	if !thinking.Exists() {
		return
	}
	tType := thinking.Get("type").String()
	if tType == "" {
		return
	}
	spec := &ir.ThinkingSpec{
		Type:    tType,
		Display: thinking.Get("display").String(),
	}
	if b := thinking.Get("budget_tokens"); b.Exists() && b.Type == gjson.Number {
		spec.BudgetTokens = b.Int()
	}
	ext.Thinking = spec
}

func captureCacheControl(blockID string, cc gjson.Result, ext *ir.IRExtensions) {
	if !cc.Exists() {
		return
	}
	ext.SetCacheControl(blockID, parseCacheControlSpec(cc))
}

func parseCacheControlSpec(cc gjson.Result) ir.CacheControlSpec {
	return ir.CacheControlSpec{
		Type: cc.Get("type").String(),
		TTL:  cc.Get("ttl").String(),
	}
}

func captureTopLevelCacheControl(body []byte, ext *ir.IRExtensions) {
	cc := gjson.GetBytes(body, "cache_control")
	if !cc.Exists() {
		return
	}
	ext.SetCacheControl("$.cache_control", parseCacheControlSpec(cc))
}

func captureThinkingBlock(blockID string, block gjson.Result, ext *ir.IRExtensions) {
	if sig := block.Get("signature").String(); sig != "" {
		ext.SetThinkingSignature(blockID, sig)
	}
}

func captureDocumentBlock(blockID string, block gjson.Result, ext *ir.IRExtensions) {
	doc := ir.DocumentBlock{BlockID: blockID}
	source := block.Get("source")
	if source.Exists() {
		doc.SourceType = source.Get("type").String()
		doc.MediaType = source.Get("media_type").String()
		switch doc.SourceType {
		case "base64", "text":
			doc.Data = source.Get("data").String()
		case "url":
			doc.Data = source.Get("url").String()
		case "file", "file_id":
			doc.Data = source.Get("file_id").String()
			if doc.Data == "" {
				doc.Data = source.Get("id").String()
			}
		case "content":
			source.Get("content").ForEach(func(idx, sub gjson.Result) bool {
				doc.InlineContent = append(doc.InlineContent, ir.SystemBlock{
					BlockID: fmt.Sprintf("%s.source.content[%d]", blockID, idx.Int()),
					Text:    sub.Get("text").String(),
				})
				return true
			})
		}
	}
	if citations := block.Get("citations.enabled"); citations.Type == gjson.True {
		doc.Citations = true
		ext.CitationsEnabled = true
	}
	if raw := block.Get("cache_control"); raw.Exists() {
		spec := parseCacheControlSpec(raw)
		doc.CacheControl = &spec
	}
	ext.Documents = append(ext.Documents, doc)
}

func captureServerToolUse(blockID string, block gjson.Result, ext *ir.IRExtensions) {
	spec := ir.ServerToolSpec{
		Type: "server_tool_use",
		Name: block.Get("name").String(),
	}
	if input := block.Get("input"); input.Exists() {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(input.Raw), &parsed); err == nil {
			spec.Parameters = parsed
		}
	}
	ext.ServerTools = append(ext.ServerTools, spec)
	ext.AppendWarning(ir.Warning{
		Field:    blockID,
		Reason:   "server_tool_use_captured",
		Severity: ir.WarningSeverityInfo,
	})
}
