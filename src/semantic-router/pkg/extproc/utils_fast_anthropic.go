package extproc

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

// extractContentFastAnthropic mirrors extractContentFast for the Anthropic
// /v1/messages request shape so the conversation-signal family computes
// equivalent fast-path signals regardless of inbound protocol.
//
// The shape differs from OpenAI in three ways that matter for fast
// extraction:
//
//  1. system is either a string or a TextBlockParam[] — when present, it
//     counts as one system message regardless of array length, and its
//     text contributes to the conversation transcript.
//  2. user / assistant content blocks carry image data under the
//     image.source object (base64 / url / file_id), not under image_url.
//  3. tool_use blocks appear inline on assistant turns; tool_result blocks
//     appear inline on user turns under the same content array.
//
// We accept the cost of walking the inbound body twice (once here, once
// in ParseAnthropicRequest in commit 4) because fast-extract has to run
// before validation, classification, and the SDK parse — short-circuiting
// here lets fast_response / rate limit / cache short-circuits avoid the
// SDK allocation in the common case.
func extractContentFastAnthropic(body []byte) (*FastExtractResult, error) {
	r := &FastExtractResult{}
	if err := extractModelAndStreamFast(body, r); err != nil {
		return nil, err
	}

	if system := gjson.GetBytes(body, "system"); system.Exists() {
		consumeFastExtractAnthropicSystem(system, r)
	}

	if messages := gjson.GetBytes(body, "messages"); messages.Exists() && messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			consumeFastExtractAnthropicMessage(msg, r)
			return true
		})
	}

	if tools := gjson.GetBytes(body, "tools"); tools.Exists() && tools.IsArray() {
		tools.ForEach(func(_, _ gjson.Result) bool {
			r.ToolDefinitionCount++
			return true
		})
	}

	return r, nil
}

func consumeFastExtractAnthropicSystem(system gjson.Result, result *FastExtractResult) {
	text := extractAnthropicSystemText(system)
	// Anthropic's system field is one logical system message regardless of
	// whether it arrived as a plain string or as an array of TextBlockParam
	// entries. Mirror that by counting it as one entry, matching how the
	// OpenAI fast extractor counts a single role=system message.
	result.SystemMessageCount++
	if text != "" {
		result.NonUserMessages = append(result.NonUserMessages, text)
	}
}

func extractAnthropicSystemText(system gjson.Result) string {
	if system.Type == gjson.String {
		return system.String()
	}
	if !system.IsArray() {
		return ""
	}
	var parts []string
	system.ForEach(func(_, block gjson.Result) bool {
		if t := block.Get("text").String(); t != "" {
			parts = append(parts, t)
		}
		return true
	})
	return strings.Join(parts, " ")
}

func consumeFastExtractAnthropicMessage(msg gjson.Result, result *FastExtractResult) {
	previousWasToolResult := result.LastMessageToolResult
	role := msg.Get("role").String()
	content := msg.Get("content")
	result.LastMessageRole = role
	result.LastMessageToolResult = false
	result.LastUserAfterToolResult = false

	text := extractAnthropicTextFromContent(content)
	hasToolResult, hasToolUse := scanAnthropicBlockTypes(content)

	switch role {
	case "user":
		result.UserMessageCount++
		result.LastUserAfterToolResult = previousWasToolResult && !hasToolResult
		recordAnthropicUserMessage(result, text, content)
		if hasToolResult {
			// Anthropic carries tool_result blocks inline on user turns;
			// in the OpenAI shape these become tool-role messages, so
			// count them on the tool channel as well to keep downstream
			// signal parity.
			result.ToolMessageCount++
			result.ToolResultCount++
			result.LastMessageToolResult = true
		}
	case "assistant":
		result.AssistantMessageCount++
		recordFastExtractNonUserMessage(result, role, text)
		if hasToolUse {
			countAnthropicAssistantToolUses(content, result)
		}
	}
}

func recordAnthropicUserMessage(result *FastExtractResult, text string, content gjson.Result) {
	if result.FirstImageURL == "" {
		result.FirstImageURL = extractAnthropicImageURLFromContent(content)
	}
	if text == "" {
		return
	}
	if result.UserContent != "" {
		result.PriorUserMessages = append(result.PriorUserMessages, result.UserContent)
	}
	result.UserContent = text
}

func countAnthropicAssistantToolUses(content gjson.Result, result *FastExtractResult) {
	if !content.IsArray() {
		return
	}
	content.ForEach(func(_, block gjson.Result) bool {
		if block.Get("type").String() != "tool_use" {
			return true
		}
		result.AssistantToolCallCount++
		if name := block.Get("name").String(); name != "" {
			result.AssistantToolNames = append(result.AssistantToolNames, name)
		}
		return true
	})
}

// extractAnthropicTextFromContent collects the text payload from an
// Anthropic content field. Plain string content (the common short-message
// shape) is returned verbatim; array content yields a space-joined string
// of every text block in document order.
func extractAnthropicTextFromContent(content gjson.Result) string {
	if content.Type == gjson.String {
		return content.String()
	}
	if !content.IsArray() {
		return ""
	}
	var parts []string
	content.ForEach(func(_, block gjson.Result) bool {
		if block.Get("type").String() != "text" {
			return true
		}
		if t := block.Get("text").String(); t != "" {
			parts = append(parts, t)
		}
		return true
	})
	if len(parts) == 1 {
		return parts[0]
	}
	return strings.Join(parts, " ")
}

func scanAnthropicBlockTypes(content gjson.Result) (hasToolResult bool, hasToolUse bool) {
	if !content.IsArray() {
		return false, false
	}
	content.ForEach(func(_, block gjson.Result) bool {
		switch block.Get("type").String() {
		case "tool_result":
			hasToolResult = true
		case "tool_use":
			hasToolUse = true
		}
		return true
	})
	return hasToolResult, hasToolUse
}

// extractAnthropicImageURLFromContent returns the first safe-to-surface
// image reference from an Anthropic content array. Only inline base64
// sources are returned (the same SSRF-safety policy as the OpenAI fast
// extractor); URL and file_id sources are silently skipped here and
// handled by the full inbound parser.
func extractAnthropicImageURLFromContent(content gjson.Result) string {
	if !content.IsArray() {
		return ""
	}
	var found string
	content.ForEach(func(_, block gjson.Result) bool {
		if block.Get("type").String() != "image" {
			return true
		}
		source := block.Get("source")
		if source.Get("type").String() != "base64" {
			return true
		}
		mediaType := source.Get("media_type").String()
		data := source.Get("data").String()
		if mediaType == "" || data == "" {
			return true
		}
		candidate := fmt.Sprintf("data:%s;base64,%s", mediaType, data)
		if isSafeImageDataURL(candidate) {
			found = candidate
			return false
		}
		return true
	})
	return found
}
