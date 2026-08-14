package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyPreDispatchToolsPolicy applies mode=none after decision evaluation but
// before provider or looper dispatch. The matched decision and its signal facts
// therefore remain stable while privacy-oriented routes avoid forwarding tool
// controls and prior tool history to the selected backend.
func (r *OpenAIRouter) applyPreDispatchToolsPolicy(
	request *openai.ChatCompletionNewParams,
	ctx *RequestContext,
) (bool, error) {
	toolsCfg := resolveDecisionToolsConfig(ctx)
	if request == nil || toolsCfg == nil || !toolsCfg.Enabled || toolsCfg.EffectiveMode() != config.ToolsPluginModeNone {
		return false, nil
	}
	if ctx != nil && ctx.ClientProtocol == config.ClientProtocolAnthropic {
		return applyAnthropicPreDispatchToolsPolicy(request, ctx, toolsCfg.StripToolHistory)
	}
	return r.applyOpenAIPreDispatchToolsPolicy(request, ctx, toolsCfg.StripToolHistory)
}

func applyAnthropicPreDispatchToolsPolicy(
	request *openai.ChatCompletionNewParams,
	ctx *RequestContext,
	stripHistory bool,
) (bool, error) {
	body, removed, err := stripAnthropicToolPolicyBody(ctx.workingRequestBody(), stripHistory)
	if err != nil {
		return false, fmt.Errorf("apply Anthropic pre-dispatch tools policy: %w", err)
	}
	reparsed, err := parseRequestForProtocol(ctx, body)
	if err != nil {
		return false, fmt.Errorf("reparse Anthropic tools policy body: %w", err)
	}
	*request = *reparsed
	ctx.setWorkingRequestBody(body)
	if removed > 0 {
		logging.Infof("[ToolsPlugin] Decision %q stripped %d prior Anthropic tool-history messages or blocks", ctx.VSRSelectedDecision.Name, removed)
	}
	return true, nil
}

func (r *OpenAIRouter) applyOpenAIPreDispatchToolsPolicy(
	request *openai.ChatCompletionNewParams,
	ctx *RequestContext,
	stripHistory bool,
) (bool, error) {
	body, removed, err := stripOpenAIToolPolicyBody(ctx.workingRequestBody(), stripHistory)
	if err != nil {
		return false, fmt.Errorf("apply OpenAI pre-dispatch tools policy: %w", err)
	}
	reparsed, err := parseRequestForProtocol(ctx, body)
	if err != nil {
		return false, fmt.Errorf("reparse OpenAI tools policy body: %w", err)
	}
	*request = *reparsed
	ctx.setWorkingRequestBody(body)
	r.refreshResponseAPITranslatedBody(ctx, body)
	if removed > 0 {
		logging.Infof("[ToolsPlugin] Decision %q stripped %d prior tool-history messages or fields", ctx.VSRSelectedDecision.Name, removed)
	}
	return true, nil
}

// stripOpenAIToolPolicyBody applies mode=none directly to the current wire
// document. Only tool controls, tool-result messages, and assistant tool-call
// fields are removed. Retained messages stay as raw JSON so provider-specific
// fields such as reasoning_content and cache_control survive the policy.
func stripOpenAIToolPolicyBody(body []byte, stripHistory bool) ([]byte, int, error) {
	if !gjson.ValidBytes(body) {
		return nil, 0, fmt.Errorf("request body is not valid JSON")
	}
	mutated, err := deleteJSONFields(body, []string{
		"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls", "web_search_options",
	})
	if err != nil {
		return nil, 0, err
	}
	if !stripHistory {
		return mutated, 0, nil
	}
	messages := gjson.GetBytes(mutated, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return mutated, 0, nil
	}
	filtered, removed, err := filterOpenAIToolHistory(messages.Array())
	if err != nil {
		return nil, 0, err
	}
	return replaceJSONMessages(mutated, filtered, removed)
}

func deleteJSONFields(body []byte, fields []string) ([]byte, error) {
	var err error
	for _, field := range fields {
		body, err = sjson.DeleteBytes(body, field)
		if err != nil {
			return nil, err
		}
	}
	return body, nil
}

func filterOpenAIToolHistory(messages []gjson.Result) ([]json.RawMessage, int, error) {
	filtered := make([]json.RawMessage, 0, len(messages))
	removed := 0
	for _, message := range messages {
		role := message.Get("role").String()
		if role == "tool" || role == "function" {
			removed++
			continue
		}

		messageRaw := []byte(message.Raw)
		if role == "assistant" && (message.Get("tool_calls").Exists() || message.Get("function_call").Exists()) {
			var keep bool
			var err error
			messageRaw, keep, err = stripOpenAIAssistantToolHistory(messageRaw)
			if err != nil {
				return nil, 0, err
			}
			removed++
			if !keep {
				continue
			}
		}
		filtered = append(filtered, json.RawMessage(messageRaw))
	}
	return filtered, removed, nil
}

func stripOpenAIAssistantToolHistory(message []byte) ([]byte, bool, error) {
	message, err := deleteJSONFields(message, []string{"tool_calls", "function_call"})
	if err != nil {
		return nil, false, err
	}
	return message, openAIAssistantHasNonToolPayload(message), nil
}

func replaceJSONMessages(body []byte, filtered []json.RawMessage, removed int) ([]byte, int, error) {
	encoded, err := json.Marshal(filtered)
	if err != nil {
		return nil, 0, err
	}
	mutated, err := sjson.SetRawBytes(body, "messages", encoded)
	if err != nil {
		return nil, 0, err
	}
	return mutated, removed, nil
}

func openAIAssistantHasNonToolPayload(message []byte) bool {
	meaningful := false
	gjson.ParseBytes(message).ForEach(func(key, value gjson.Result) bool {
		switch key.String() {
		case "role", "name", "tool_calls", "function_call":
			return true
		default:
			if value.Type != gjson.Null {
				meaningful = true
				return false
			}
			return true
		}
	})
	return meaningful
}

// stripAnthropicToolPolicyBody applies mode=none to the native Anthropic body
// before rebuilding its canonical IR. Keeping the native document and the IR
// in lockstep prevents message-indexed image/cache-control passthrough data
// from drifting after historical tool blocks are removed.
func stripAnthropicToolPolicyBody(body []byte, stripHistory bool) ([]byte, int, error) {
	if !gjson.ValidBytes(body) {
		return nil, 0, fmt.Errorf("request body is not valid JSON")
	}
	mutated, err := deleteJSONFields(body, []string{"tools", "tool_choice"})
	if err != nil {
		return nil, 0, err
	}
	if !stripHistory {
		return mutated, 0, nil
	}

	messages := gjson.GetBytes(mutated, "messages").Array()
	filtered, removed, err := filterAnthropicToolHistory(messages)
	if err != nil {
		return nil, 0, err
	}
	return replaceJSONMessages(mutated, filtered, removed)
}

func filterAnthropicToolHistory(messages []gjson.Result) ([]json.RawMessage, int, error) {
	filtered := make([]json.RawMessage, 0, len(messages))
	removed := 0
	for _, message := range messages {
		messageRaw, messageRemoved, keep, err := stripAnthropicMessageToolBlocks(message)
		if err != nil {
			return nil, 0, err
		}
		removed += messageRemoved
		if !keep {
			continue
		}
		filtered = append(filtered, json.RawMessage(messageRaw))
	}
	return filtered, removed, nil
}

func stripAnthropicMessageToolBlocks(message gjson.Result) ([]byte, int, bool, error) {
	role := message.Get("role").String()
	content := message.Get("content")
	messageRaw := []byte(message.Raw)
	if (role != "assistant" && role != "user") || !content.IsArray() {
		return messageRaw, 0, true, nil
	}
	kept, removed := filterAnthropicContentBlocks(role, content.Array())
	if len(kept) == 0 {
		return nil, removed, false, nil
	}
	encoded, err := json.Marshal(kept)
	if err != nil {
		return nil, 0, false, err
	}
	messageRaw, err = sjson.SetRawBytes(messageRaw, "content", encoded)
	if err != nil {
		return nil, 0, false, err
	}
	return messageRaw, removed, true, nil
}

func filterAnthropicContentBlocks(role string, blocks []gjson.Result) ([]json.RawMessage, int) {
	kept := make([]json.RawMessage, 0, len(blocks))
	removed := 0
	for _, block := range blocks {
		if isAnthropicToolHistoryBlock(role, block.Get("type").String()) {
			removed++
			continue
		}
		kept = append(kept, json.RawMessage(block.Raw))
	}
	return kept, removed
}

func isAnthropicToolHistoryBlock(role, blockType string) bool {
	if role == "user" {
		return blockType == "tool_result"
	}
	return role == "assistant" && (blockType == "tool_use" || blockType == "server_tool_use")
}

func applyToolsPluginRequestPolicy(
	request *openai.ChatCompletionNewParams,
	toolsCfg *config.ToolsPluginConfig,
) int {
	if request == nil || toolsCfg == nil || !toolsCfg.Enabled {
		return 0
	}
	if toolsCfg.EffectiveMode() == config.ToolsPluginModeNone {
		clearTypedToolControls(request)
	}
	if !toolsCfg.StripToolHistory {
		return 0
	}

	filtered, removed := filterTypedToolHistory(request.Messages)
	if removed > 0 {
		request.Messages = filtered
	}
	return removed
}

func clearTypedToolControls(request *openai.ChatCompletionNewParams) {
	request.Tools = nil
	request.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	request.Functions = nil
	request.FunctionCall = openai.ChatCompletionNewParamsFunctionCallUnion{}
	request.ParallelToolCalls = param.Opt[bool]{}
	request.WebSearchOptions = openai.ChatCompletionNewParamsWebSearchOptions{}
}

func filterTypedToolHistory(
	messages []openai.ChatCompletionMessageParamUnion,
) ([]openai.ChatCompletionMessageParamUnion, int) {
	filtered := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	removed := 0
	for _, message := range messages {
		if message.OfTool != nil || message.OfFunction != nil {
			removed++
			continue
		}
		if message.OfAssistant != nil && assistantHasToolHistory(message.OfAssistant) {
			assistant := *message.OfAssistant
			assistant.ToolCalls = nil
			assistant.FunctionCall = openai.ChatCompletionAssistantMessageParamFunctionCall{}
			removed++
			if !assistantHasNonToolPayload(&assistant) {
				continue
			}
			message.OfAssistant = &assistant
		}
		filtered = append(filtered, message)
	}
	return filtered, removed
}

func assistantHasToolHistory(assistant *openai.ChatCompletionAssistantMessageParam) bool {
	if len(assistant.ToolCalls) > 0 {
		return true
	}
	legacyCall := assistant.FunctionCall
	return legacyCall.Name != "" || legacyCall.Arguments != ""
}

func assistantHasNonToolPayload(assistant *openai.ChatCompletionAssistantMessageParam) bool {
	return assistant.Content.OfString.Valid() ||
		len(assistant.Content.OfArrayOfContentParts) > 0 ||
		assistant.Refusal.Valid() ||
		assistant.Audio.ID != ""
}
