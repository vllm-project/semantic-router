package classification

import (
	"bytes"
	"encoding/json"
	"strings"
)

// ToolChoiceFacts captures protocol-level tool execution constraints without
// retaining the request payload. Required and None are mutually exclusive.
type ToolChoiceFacts struct {
	Required bool
	None     bool
}

// ResolveOpenAIToolChoiceFacts interprets Chat Completions tool_choice and the
// legacy function_call field. When tool_choice is present it is authoritative,
// including when its value is unknown or malformed.
func ResolveOpenAIToolChoiceFacts(toolChoice, functionCall json.RawMessage) ToolChoiceFacts {
	if rawJSONPresent(toolChoice) {
		return resolveOpenAIToolChoice(toolChoice)
	}
	if rawJSONPresent(functionCall) {
		return resolveLegacyFunctionCall(functionCall)
	}
	return ToolChoiceFacts{}
}

// ResolveAnthropicToolChoiceFacts interprets the Messages API tool_choice
// object. "any" and a named "tool" require execution; "none" forbids it.
func ResolveAnthropicToolChoiceFacts(toolChoice json.RawMessage) ToolChoiceFacts {
	if !rawJSONPresent(toolChoice) {
		return ToolChoiceFacts{}
	}
	var object struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(toolChoice, &object); err != nil {
		return ToolChoiceFacts{}
	}
	switch object.Type {
	case "any":
		return ToolChoiceFacts{Required: true}
	case "tool":
		return ToolChoiceFacts{Required: strings.TrimSpace(object.Name) != ""}
	case "none":
		return ToolChoiceFacts{None: true}
	default:
		return ToolChoiceFacts{}
	}
}

func resolveOpenAIToolChoice(raw json.RawMessage) ToolChoiceFacts {
	var mode string
	if err := json.Unmarshal(raw, &mode); err == nil {
		switch mode {
		case "required":
			return ToolChoiceFacts{Required: true}
		case "none":
			return ToolChoiceFacts{None: true}
		default:
			return ToolChoiceFacts{}
		}
	}

	var object struct {
		Type     string            `json:"type"`
		Name     string            `json:"name"`
		Mode     string            `json:"mode"`
		Tools    []json.RawMessage `json:"tools"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return ToolChoiceFacts{}
	}
	if object.Type == "function" &&
		(strings.TrimSpace(object.Name) != "" || strings.TrimSpace(object.Function.Name) != "") {
		return ToolChoiceFacts{Required: true}
	}
	if object.Type == "allowed_tools" && object.Mode == "required" && len(object.Tools) > 0 {
		return ToolChoiceFacts{Required: true}
	}
	return ToolChoiceFacts{}
}

func resolveLegacyFunctionCall(raw json.RawMessage) ToolChoiceFacts {
	var mode string
	if err := json.Unmarshal(raw, &mode); err == nil {
		if mode == "none" {
			return ToolChoiceFacts{None: true}
		}
		return ToolChoiceFacts{}
	}
	var object struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &object); err == nil && strings.TrimSpace(object.Name) != "" {
		return ToolChoiceFacts{Required: true}
	}
	return ToolChoiceFacts{}
}

func rawJSONPresent(raw json.RawMessage) bool {
	return len(bytes.TrimSpace(raw)) > 0
}
