package protocolcodec

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenAI custom tools take free-form input instead of JSON arguments. Chat
// nests the definition and each call under "custom".
type customToolFormatWire struct {
	Type    string                 `json:"type"`
	Grammar *customToolGrammarWire `json:"grammar,omitempty"`
}

type customToolGrammarWire struct {
	Definition string `json:"definition"`
	Syntax     string `json:"syntax"`
}

type chatCustomToolWire struct {
	Name        string                `json:"name"`
	Description string                `json:"description,omitempty"`
	Format      *customToolFormatWire `json:"format,omitempty"`
}

type chatCustomCallWire struct {
	Name  string `json:"name,omitempty"`
	Input string `json:"input"`
}

func decodeCustomToolFormat(wire *customToolFormatWire) (*llmprotocol.CustomToolFormat, error) {
	if wire == nil {
		return nil, nil
	}
	switch wire.Type {
	case "text":
		if wire.Grammar != nil {
			return nil, invalidCustomToolFormat("custom tool text format cannot carry a grammar")
		}
		return nil, nil
	case "grammar":
		if wire.Grammar == nil {
			return nil, invalidCustomToolFormat("custom tool grammar format requires a grammar")
		}
		return &llmprotocol.CustomToolFormat{Syntax: wire.Grammar.Syntax, Definition: wire.Grammar.Definition}, nil
	default:
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_format", "custom tool format must be text or grammar", nil,
		)
	}
}

func invalidCustomToolFormat(message string) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_format", message, nil)
}

func encodeCustomToolFormat(format *llmprotocol.CustomToolFormat) *customToolFormatWire {
	if format == nil {
		return nil
	}
	return &customToolFormatWire{
		Type:    "grammar",
		Grammar: &customToolGrammarWire{Definition: format.Definition, Syntax: format.Syntax},
	}
}

func decodeChatCustomTool(wire chatToolWire) (llmprotocol.Tool, error) {
	function := wire.Function
	if wire.Custom == nil || function.Name != "" || function.Description != "" || len(function.Parameters) != 0 || function.Strict != nil {
		return llmprotocol.Tool{}, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "invalid_tool_variant", "Chat custom tools carry only a custom definition", nil,
		)
	}
	format, err := decodeCustomToolFormat(wire.Custom.Format)
	if err != nil {
		return llmprotocol.Tool{}, err
	}
	return llmprotocol.Tool{
		Kind: llmprotocol.ToolKindCustom, Name: wire.Custom.Name, Description: wire.Custom.Description,
		CustomFormat: format, Cache: decodeAnthropicCacheControl(wire.CacheControl),
	}, nil
}

func encodeChatCustomTool(tool llmprotocol.Tool) chatToolWire {
	return chatToolWire{
		Type: "custom",
		Custom: &chatCustomToolWire{
			Name: tool.Name, Description: tool.Description, Format: encodeCustomToolFormat(tool.CustomFormat),
		},
		CacheControl: encodeAnthropicCacheControl(tool.Cache),
	}
}

func decodeChatToolCall(wire chatToolCallWire) (llmprotocol.ToolCall, error) {
	if wire.Type == "custom" {
		if wire.Custom == nil || wire.Function != (chatFunctionCallWire{}) {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "Chat custom tool calls carry only a custom call", nil,
			)
		}
		return llmprotocol.ToolCall{
			Kind: llmprotocol.ToolKindCustom, ID: wire.ID, Name: wire.Custom.Name, Arguments: wire.Custom.Input,
		}, nil
	}
	if (wire.Type != "" && wire.Type != "function") || wire.Custom != nil {
		return llmprotocol.ToolCall{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_call", "only function and custom tool calls enter the model protocol", nil,
		)
	}
	return llmprotocol.ToolCall{ID: wire.ID, Name: wire.Function.Name, Arguments: wire.Function.Arguments}, nil
}

func encodeChatToolCall(call llmprotocol.ToolCall) chatToolCallWire {
	if call.Kind == llmprotocol.ToolKindCustom {
		return chatToolCallWire{ID: call.ID, Type: "custom", Custom: &chatCustomCallWire{Name: call.Name, Input: call.Arguments}}
	}
	return chatToolCallWire{
		ID: call.ID, Type: "function",
		Function: chatFunctionCallWire{Name: call.Name, Arguments: call.Arguments},
	}
}

// A streamed custom call names itself once and then sends input fragments
// under "custom" alone, so the type is only required on the first delta.
func decodeChatToolCallDelta(wire chatChunkToolCallWire) (llmprotocol.ToolCall, error) {
	if wire.Type == "custom" || wire.Custom != nil {
		if wire.Type != "" && wire.Type != "custom" || wire.Custom == nil || wire.Function != (chatFunctionCallWire{}) {
			return llmprotocol.ToolCall{}, invalidProviderResponse("invalid_stream_tool_call", "Chat stream custom tool call delta is invalid")
		}
		return llmprotocol.ToolCall{
			Kind: llmprotocol.ToolKindCustom, ID: wire.ID, Name: wire.Custom.Name, Arguments: wire.Custom.Input,
		}, nil
	}
	if wire.Type != "" && wire.Type != "function" {
		return llmprotocol.ToolCall{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_call", "only function and custom tool calls enter the model protocol", nil,
		)
	}
	return llmprotocol.ToolCall{ID: wire.ID, Name: wire.Function.Name, Arguments: wire.Function.Arguments}, nil
}
