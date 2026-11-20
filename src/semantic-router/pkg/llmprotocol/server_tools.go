package llmprotocol

import (
	"bytes"
	"encoding/json"
	"strings"
)

// UsesServerTools reports whether a request declares a tool its provider
// defines, or holds a block such a tool left in an earlier turn.
func UsesServerTools(request Request) bool {
	return RequiredCapabilities(request).Supports(CapabilityServerTools)
}

// IsServerToolContent reports whether a block is a server tool's call or
// result, or text carrying the citations a server tool attached to it.
func IsServerToolContent(content Content) bool {
	return serverToolContentCapability(content) != 0
}

func validateServerTool(tool Tool, limits Limits) error {
	if strings.TrimSpace(tool.Name) == "" || len(tool.InputSchema) != 0 || tool.Strict != nil ||
		tool.CustomFormat != nil || tool.Description != "" {
		return NewError(ErrorInvalidRequest, "invalid_tool", "a server tool needs a name and keeps the rest of its definition as sent", nil)
	}
	if exceeds(tool.Name, limits.ToolNameBytes) {
		return NewError(ErrorInvalidRequest, "tool_text_limit", "tool name exceeds the configured limit", nil)
	}
	if ValidateJSONObject(tool.Raw, limits.JSONDepth) != nil {
		return NewError(ErrorInvalidRequest, "invalid_tool", "a server tool definition must be a JSON object", nil)
	}
	return validateCacheDirective(tool.Cache)
}

// validateRawContent keeps Raw to the two kinds that carry it: a server tool
// block, which holds nothing else, and text, whose citations it is.
func validateRawContent(content Content, limits Limits) error {
	switch {
	case content.Kind == ContentServerTool:
		if content.Text != "" || len(content.Citations) != 0 || content.Cache != nil || hasNonTextFields(content) ||
			content.Signature != "" || ValidateJSONObject(content.Raw, limits.JSONDepth) != nil {
			return NewError(ErrorInvalidRequest, "invalid_content", "a server tool block holds only its provider block", nil)
		}
	case len(content.Raw) == 0:
	case content.Kind != ContentText || !isJSONArray(content.Raw):
		return NewError(ErrorInvalidRequest, "invalid_content", "only text may carry server tool citations, as a JSON array", nil)
	}
	return nil
}

func isJSONArray(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '[' && json.Valid(trimmed)
}

// ProjectServerToolHistory sends a backend that cannot read server tool
// blocks the conversation without them: each call and result is dropped and
// cited text keeps its text without the citations. A non-Anthropic model could
// not read the results, which Anthropic encrypts, and the answer that used
// them stays in the history.
func ProjectServerToolHistory(request Request, target WireFormat) (Request, Diagnostics) {
	dropped := false
	messages := make([]Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]Content, 0, len(message.Content))
		for _, content := range message.Content {
			switch {
			case !IsServerToolContent(content):
			case content.Kind == ContentText:
				content.Raw, dropped = nil, true
			default:
				dropped = true
				continue
			}
			contents = append(contents, content)
		}
		if len(contents) == 0 && message.ReasoningEffort == "" {
			continue
		}
		message.Content = contents
		messages = append(messages, message)
	}
	if !dropped {
		return request, nil
	}
	projected := request
	projected.Messages = messages
	projected.Generation++ // The source envelope still carries the blocks.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat, Target: target, Field: "content.server_tool", Action: DiagnosticDropped,
		Reason: "server tool calls, results and citations left out for a backend that cannot read them",
	}}
}

// AppendCitations adds the citations a streamed text delta carries to those
// its block already has. Both are JSON arrays.
func AppendCitations(current, next json.RawMessage) json.RawMessage {
	if len(next) == 0 {
		return current
	}
	var left, right []json.RawMessage
	_ = json.Unmarshal(current, &left)
	if json.Unmarshal(next, &right) != nil {
		return current
	}
	joined, _ := json.Marshal(append(left, right...))
	return joined
}
