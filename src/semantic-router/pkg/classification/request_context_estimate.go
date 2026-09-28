package classification

import (
	"math/big"
	"strings"

	"github.com/tidwall/gjson"
)

const (
	// RequestContextBytesPerToken deliberately matches the context signal's
	// cold-start byte heuristic. Keeping the prose conversion shared minimizes
	// context-band drift; only the newly explicit role/framing reserve is added.
	RequestContextBytesPerToken = CharactersPerToken

	// RequestContextStructuredBytesPerToken intentionally assumes as little as
	// one byte per token for schemas, tool arguments, and other structured JSON.
	// Punctuation-heavy payloads tokenize much more densely than prose, so the
	// prose 4B/token heuristic is not a safe routing-admission lower bound.
	RequestContextStructuredBytesPerToken = 1

	// RequestContextImageTokenBudget is a provider-neutral routing reserve, not
	// a tokenizer-exact vision charge. Image preprocessing varies by provider,
	// resolution, tiling, and detail mode; reserving 8K tokens per image is a
	// deliberately conservative admission estimate while keeping image URLs and
	// base64 payloads out of text-derived signals and diagnostics.
	RequestContextImageTokenBudget = 8 * 1024

	// Provider chat templates add role/control tokens that are not present in
	// message content. These fixed reserves are intentionally provider-neutral;
	// they complement (rather than duplicate) counted role/name/id text.
	RequestContextMessageFramingTokens        = 4
	RequestContextToolCallFramingTokens       = 8
	RequestContextToolDefinitionFramingTokens = 8
)

// RequestContextEstimate is a content-free summary of the request components
// that can consume a model context window. TextBytes and StructuredBytes cover
// every message role, structured message content, assistant tool calls, tool
// results, and top-level tool/function schemas. Image payload bytes are
// excluded and represented by ImageCount and the fixed image-token reserve.
//
// TokenFloor is a conservative routing floor. EquivalentBytes is its
// content-free byte equivalent for bounded diagnostics and bookkeeping; it is
// deliberately excluded from prose-token calibration. Neither field is
// intended to reproduce a provider tokenizer exactly.
//
// The admission formula is:
//
//	ceil(TextBytes / 4) + StructuredBytes + 8192*ImageCount +
//	FramingTokens + OutputTokenReserve
type RequestContextEstimate struct {
	TextBytes           int
	StructuredBytes     int
	ImageCount          int
	MessageCount        int
	ToolCallCount       int
	ToolDefinitionCount int
	FramingTokens       int
	OutputTokenReserve  int
	HasNonText          bool
	TokenFloor          int
	EquivalentBytes     int
}

type requestContextAccumulator struct {
	textBytes           int
	structuredBytes     int
	imageCount          int
	messageCount        int
	toolCallCount       int
	toolDefinitionCount int
	framingTokens       int
	outputTokenReserve  int
	hasNonText          bool
}

// EstimateOpenAIRequestContext estimates the context represented by an
// OpenAI-compatible Chat Completions envelope. The walker consumes raw JSON so
// structured values, including integers larger than 2^53, retain their exact
// lexical size instead of passing through float64.
func EstimateOpenAIRequestContext(body []byte) RequestContextEstimate {
	var estimate requestContextAccumulator

	messages := gjson.GetBytes(body, "messages")
	if messages.Exists() {
		if messages.IsArray() {
			messages.ForEach(func(_, message gjson.Result) bool {
				estimate.addOpenAIMessage(message)
				return true
			})
		} else {
			estimate.addStructuredJSON(messages)
		}
	}

	estimate.addToolDefinitions(gjson.GetBytes(body, "tools"))
	estimate.addToolDefinitions(gjson.GetBytes(body, "functions"))
	estimate.addStructuredJSON(gjson.GetBytes(body, "tool_choice"))
	estimate.addStructuredJSON(gjson.GetBytes(body, "function_call"))
	estimate.addStructuredJSON(gjson.GetBytes(body, "response_format"))
	estimate.addEffectiveOutputReserve(
		gjson.GetBytes(body, "max_completion_tokens"),
		gjson.GetBytes(body, "max_tokens"),
	)

	return estimate.result()
}

// EstimateAnthropicRequestContext estimates the context represented by an
// Anthropic Messages envelope using the same byte and image budgets as the
// OpenAI-compatible path.
func EstimateAnthropicRequestContext(body []byte) RequestContextEstimate {
	var estimate requestContextAccumulator

	if system := gjson.GetBytes(body, "system"); system.Exists() {
		estimate.addSyntheticMessageRole("system")
		estimate.addAnthropicContent(system)
	}

	estimate.addAnthropicMessages(gjson.GetBytes(body, "messages"))

	estimate.addToolDefinitions(gjson.GetBytes(body, "tools"))
	estimate.addStructuredJSON(gjson.GetBytes(body, "tool_choice"))
	estimate.addEffectiveOutputReserve(gjson.GetBytes(body, "max_tokens"))
	return estimate.result()
}

func (estimate *requestContextAccumulator) addAnthropicMessages(messages gjson.Result) {
	if !messages.Exists() {
		return
	}
	if !messages.IsArray() {
		estimate.addStructuredJSON(messages)
		return
	}
	messages.ForEach(func(_, message gjson.Result) bool {
		estimate.addAnthropicMessage(message)
		return true
	})
}

func (estimate *requestContextAccumulator) addAnthropicMessage(message gjson.Result) {
	if !message.IsObject() {
		estimate.messageCount = saturatingAddContextEstimate(estimate.messageCount, 1)
		estimate.addFramingTokens(RequestContextMessageFramingTokens)
		estimate.addStructuredJSON(message)
		return
	}
	estimate.addMessageEnvelope(message)
	estimate.addAnthropicContent(message.Get("content"))
}

func (estimate *requestContextAccumulator) addOpenAIMessage(message gjson.Result) {
	if !message.IsObject() {
		estimate.messageCount = saturatingAddContextEstimate(estimate.messageCount, 1)
		estimate.addFramingTokens(RequestContextMessageFramingTokens)
		estimate.addStructuredJSON(message)
		return
	}
	estimate.addMessageEnvelope(message)
	role := strings.TrimSpace(message.Get("role").String())
	if strings.EqualFold(role, "tool") || strings.EqualFold(role, "function") {
		estimate.addOpenAIToolResultContent(message.Get("content"))
	} else {
		estimate.addOpenAIContent(message.Get("content"))
	}
	estimate.addTextResult(message.Get("refusal"))
	estimate.addTextResult(message.Get("reasoning_content"))
	estimate.addOpenAIToolCalls(message.Get("tool_calls"))
	estimate.addOpenAIFunctionCall(message.Get("function_call"))
}

func (estimate *requestContextAccumulator) addOpenAIToolResultContent(content gjson.Result) {
	estimate.hasNonText = true
	estimate.addFramingTokens(RequestContextToolCallFramingTokens)
	if !content.Exists() || content.Type == gjson.Null {
		return
	}
	if !content.IsArray() {
		if content.IsObject() && isContextImageType(content) {
			estimate.addImage()
			return
		}
		estimate.addStructuredJSON(content)
		return
	}
	content.ForEach(func(_, part gjson.Result) bool {
		if part.IsObject() {
			switch strings.ToLower(strings.TrimSpace(part.Get("type").String())) {
			case "image", "image_url", "input_image":
				estimate.addImage()
				return true
			}
		}
		estimate.addStructuredJSON(part)
		return true
	})
}

func (estimate *requestContextAccumulator) addOpenAIContent(content gjson.Result) {
	if !content.Exists() || content.Type == gjson.Null {
		return
	}
	if content.Type == gjson.String {
		estimate.addTextBytes(len(content.String()))
		return
	}
	if content.IsObject() {
		estimate.addOpenAIContentPart(content)
		return
	}
	if !content.IsArray() {
		estimate.addStructuredJSON(content)
		return
	}

	content.ForEach(func(_, part gjson.Result) bool {
		estimate.addOpenAIContentPart(part)
		return true
	})
}

func (estimate *requestContextAccumulator) addOpenAIContentPart(part gjson.Result) {
	if !part.IsObject() {
		estimate.addStructuredJSON(part)
		return
	}
	switch strings.ToLower(strings.TrimSpace(part.Get("type").String())) {
	case "image", "image_url", "input_image":
		estimate.addImage()
	case "text", "input_text", "output_text":
		text := part.Get("text")
		if text.Exists() {
			estimate.addTextResult(text)
		} else {
			estimate.addStructuredJSON(part)
		}
	default:
		// Unknown structured content still consumes prompt context. Count its
		// exact raw representation instead of silently dropping it.
		estimate.addStructuredJSON(part)
	}
}

func (estimate *requestContextAccumulator) addOpenAIToolCalls(toolCalls gjson.Result) {
	if !toolCalls.Exists() || toolCalls.Type == gjson.Null {
		return
	}
	if !toolCalls.IsArray() {
		estimate.addStructuredJSON(toolCalls)
		return
	}
	toolCalls.ForEach(func(_, call gjson.Result) bool {
		function := call.Get("function")
		if !function.Exists() || !function.IsObject() {
			estimate.toolCallCount = saturatingAddContextEstimate(estimate.toolCallCount, 1)
			estimate.addFramingTokens(RequestContextToolCallFramingTokens)
			// Count the malformed call exactly once as structured data rather
			// than separately recounting its id/type fields.
			estimate.addStructuredJSON(call)
			return true
		}
		estimate.addToolCallEnvelope(call)
		estimate.addTextResult(function.Get("name"))
		estimate.addStructuredJSON(function.Get("arguments"))
		return true
	})
}

func (estimate *requestContextAccumulator) addOpenAIFunctionCall(function gjson.Result) {
	if !function.Exists() || function.Type == gjson.Null {
		return
	}
	if !function.IsObject() {
		estimate.addStructuredJSON(function)
		return
	}
	estimate.hasNonText = true
	estimate.toolCallCount = saturatingAddContextEstimate(estimate.toolCallCount, 1)
	estimate.addFramingTokens(RequestContextToolCallFramingTokens)
	estimate.addTextResult(function.Get("name"))
	estimate.addStructuredJSON(function.Get("arguments"))
}

func (estimate *requestContextAccumulator) addAnthropicContent(content gjson.Result) {
	if !content.Exists() || content.Type == gjson.Null {
		return
	}
	if content.Type == gjson.String {
		estimate.addTextBytes(len(content.String()))
		return
	}
	if content.IsObject() {
		estimate.addAnthropicContentBlock(content)
		return
	}
	if !content.IsArray() {
		estimate.addStructuredJSON(content)
		return
	}

	content.ForEach(func(_, block gjson.Result) bool {
		estimate.addAnthropicContentBlock(block)
		return true
	})
}

func (estimate *requestContextAccumulator) addAnthropicContentBlock(block gjson.Result) {
	if !block.IsObject() {
		estimate.addStructuredJSON(block)
		return
	}
	switch strings.ToLower(strings.TrimSpace(block.Get("type").String())) {
	case "image":
		estimate.addImage()
	case "text", "input_text", "output_text":
		text := block.Get("text")
		if text.Exists() {
			estimate.addTextResult(text)
		} else {
			estimate.addStructuredJSON(block)
		}
	case "tool_use":
		estimate.addToolCallEnvelope(block)
		estimate.addTextResult(block.Get("name"))
		estimate.addStructuredJSON(block.Get("input"))
	case "tool_result":
		estimate.hasNonText = true
		estimate.addFramingTokens(RequestContextToolCallFramingTokens)
		estimate.addTextResult(block.Get("tool_use_id"))
		estimate.addAnthropicToolResultContent(block.Get("content"))
	default:
		estimate.addStructuredJSON(block)
	}
}

func (estimate *requestContextAccumulator) addAnthropicToolResultContent(content gjson.Result) {
	if !content.Exists() || content.Type == gjson.Null {
		return
	}
	if !content.IsArray() {
		if content.IsObject() && isContextImageType(content) {
			estimate.addImage()
			return
		}
		estimate.addStructuredJSON(content)
		return
	}
	content.ForEach(func(_, block gjson.Result) bool {
		if block.IsObject() && strings.EqualFold(
			strings.TrimSpace(block.Get("type").String()),
			"image",
		) {
			estimate.addImage()
			return true
		}
		estimate.addStructuredJSON(block)
		return true
	})
}

func isContextImageType(value gjson.Result) bool {
	switch strings.ToLower(strings.TrimSpace(value.Get("type").String())) {
	case "image", "image_url", "input_image":
		return true
	default:
		return false
	}
}

func (estimate *requestContextAccumulator) addTextResult(value gjson.Result) {
	if !value.Exists() || value.Type == gjson.Null {
		return
	}
	if value.Type == gjson.String {
		estimate.addTextBytes(len(value.String()))
		return
	}
	estimate.addStructuredJSON(value)
}

func (estimate *requestContextAccumulator) addMessageEnvelope(message gjson.Result) {
	estimate.messageCount = saturatingAddContextEstimate(estimate.messageCount, 1)
	estimate.addFramingTokens(RequestContextMessageFramingTokens)
	estimate.addTextResult(message.Get("role"))
	estimate.addTextResult(message.Get("name"))
	estimate.addTextResult(message.Get("tool_call_id"))
}

func (estimate *requestContextAccumulator) addSyntheticMessageRole(role string) {
	estimate.messageCount = saturatingAddContextEstimate(estimate.messageCount, 1)
	estimate.addFramingTokens(RequestContextMessageFramingTokens)
	estimate.addTextBytes(len(role))
}

func (estimate *requestContextAccumulator) addToolCallEnvelope(call gjson.Result) {
	estimate.hasNonText = true
	estimate.toolCallCount = saturatingAddContextEstimate(estimate.toolCallCount, 1)
	estimate.addFramingTokens(RequestContextToolCallFramingTokens)
	estimate.addTextResult(call.Get("id"))
	estimate.addTextResult(call.Get("type"))
	estimate.addTextResult(call.Get("tool_call_id"))
}

func (estimate *requestContextAccumulator) addToolDefinitions(definitions gjson.Result) {
	if !definitions.Exists() || definitions.Type == gjson.Null {
		return
	}
	if !definitions.IsArray() {
		estimate.addStructuredJSON(definitions)
		return
	}
	definitions.ForEach(func(_, definition gjson.Result) bool {
		estimate.toolDefinitionCount = saturatingAddContextEstimate(
			estimate.toolDefinitionCount,
			1,
		)
		estimate.addFramingTokens(RequestContextToolDefinitionFramingTokens)
		// Count each schema exactly once. Its function name, description,
		// parameters, and exact numeric literals are all inside this value.
		estimate.addStructuredJSON(definition)
		return true
	})
}

func (estimate *requestContextAccumulator) addEffectiveOutputReserve(values ...gjson.Result) {
	for _, value := range values {
		if !value.Exists() || value.Type == gjson.Null {
			continue
		}
		reserve, ok := exactNonNegativeJSONInteger(value)
		if !ok {
			// A malformed value is normally rejected before routing. A focused
			// caller still gets conservative structured accounting instead of a
			// silently missing component.
			estimate.addStructuredJSON(value)
			return
		}
		estimate.outputTokenReserve = reserve
		return
	}
}

func exactNonNegativeJSONInteger(value gjson.Result) (int, bool) {
	raw := strings.TrimSpace(value.Raw)
	if raw == "" || strings.ContainsAny(raw, ".eE") {
		return 0, false
	}
	parsed := new(big.Int)
	if _, ok := parsed.SetString(raw, 10); !ok || parsed.Sign() < 0 {
		return 0, false
	}
	maxIntValue := maxContextEstimateInt()
	maxInt := new(big.Int).SetInt64(int64(maxIntValue))
	if parsed.Cmp(maxInt) > 0 {
		return maxIntValue, true
	}
	return int(parsed.Int64()), true
}

func (estimate *requestContextAccumulator) addStructuredJSON(value gjson.Result) {
	if !value.Exists() || value.Type == gjson.Null {
		return
	}
	raw := strings.TrimSpace(value.Raw)
	if raw == "" {
		raw = value.String()
	}
	// Count the raw lexical representation. This retains escaped JSON strings
	// (the common tool-arguments form) and exact integer width without a decode
	// through float64. Whitespace is deliberately retained as a conservative
	// fallback for malformed or provider-specific structured values.
	estimate.addStructuredBytes(len(raw))
}

func (estimate *requestContextAccumulator) addTextBytes(count int) {
	estimate.textBytes = saturatingAddContextEstimate(estimate.textBytes, count)
}

func (estimate *requestContextAccumulator) addStructuredBytes(count int) {
	if count > 0 {
		estimate.hasNonText = true
	}
	estimate.structuredBytes = saturatingAddContextEstimate(estimate.structuredBytes, count)
}

func (estimate *requestContextAccumulator) addFramingTokens(count int) {
	estimate.framingTokens = saturatingAddContextEstimate(estimate.framingTokens, count)
}

func (estimate *requestContextAccumulator) addImage() {
	estimate.hasNonText = true
	if estimate.imageCount < maxContextEstimateInt() {
		estimate.imageCount++
	}
}

func (estimate requestContextAccumulator) result() RequestContextEstimate {
	textTokens := ceilContextEstimateDivision(
		estimate.textBytes,
		RequestContextBytesPerToken,
	)
	structuredTokens := ceilContextEstimateDivision(
		estimate.structuredBytes,
		RequestContextStructuredBytesPerToken,
	)
	imageTokens := saturatingMultiplyContextEstimate(
		estimate.imageCount,
		RequestContextImageTokenBudget,
	)
	tokenFloor := saturatingAddContextEstimate(textTokens, structuredTokens)
	tokenFloor = saturatingAddContextEstimate(tokenFloor, imageTokens)
	tokenFloor = saturatingAddContextEstimate(tokenFloor, estimate.framingTokens)
	tokenFloor = saturatingAddContextEstimate(tokenFloor, estimate.outputTokenReserve)
	return RequestContextEstimate{
		TextBytes:           estimate.textBytes,
		StructuredBytes:     estimate.structuredBytes,
		ImageCount:          estimate.imageCount,
		MessageCount:        estimate.messageCount,
		ToolCallCount:       estimate.toolCallCount,
		ToolDefinitionCount: estimate.toolDefinitionCount,
		FramingTokens:       estimate.framingTokens,
		OutputTokenReserve:  estimate.outputTokenReserve,
		HasNonText:          estimate.hasNonText,
		TokenFloor:          tokenFloor,
		EquivalentBytes:     saturatingMultiplyContextEstimate(tokenFloor, RequestContextBytesPerToken),
	}
}

func ceilContextEstimateDivision(value, divisor int) int {
	if value <= 0 || divisor <= 0 {
		return 0
	}
	return 1 + (value-1)/divisor
}

func saturatingAddContextEstimate(left, right int) int {
	if right <= 0 {
		return left
	}
	maxInt := maxContextEstimateInt()
	if left > maxInt-right {
		return maxInt
	}
	return left + right
}

func saturatingMultiplyContextEstimate(left, right int) int {
	if left <= 0 || right <= 0 {
		return 0
	}
	maxInt := maxContextEstimateInt()
	if left > maxInt/right {
		return maxInt
	}
	return left * right
}

func maxContextEstimateInt() int {
	return int(^uint(0) >> 1)
}
