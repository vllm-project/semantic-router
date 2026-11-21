package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesNamespaceToolWire struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Tools       []json.RawMessage `json:"tools"`
}

// decodeResponsesNamespaceTool flattens a namespace into the neutral tool list.
// Every backend receives flat names, including a compatible Responses backend,
// so one mapping serves all three formats and no backend needs namespace
// support. A request that keeps namespaces, for an OpenAI-hosted backend,
// encodes them back.
func decodeResponsesNamespaceTool(body json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy, diagnostics *llmprotocol.Diagnostics) error {
	var wire responsesNamespaceToolWire
	if err := decodeWireValue(body, &wire, policy); err != nil {
		return err
	}
	if strings.TrimSpace(wire.Name) == "" || len(wire.Tools) == 0 {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses namespace requires a name and tools", nil)
	}
	for _, entry := range wire.Tools {
		if err := decodeResponsesNamespaceEntry(entry, wire, request, policy, diagnostics); err != nil {
			return err
		}
	}
	request.Generation++
	return nil
}

func decodeResponsesNamespaceEntry(
	body json.RawMessage,
	namespace responsesNamespaceToolWire,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) error {
	toolType, err := responsesToolDiscriminator(body)
	if err != nil {
		return err
	}
	if err = validateResponsesToolVariant(body, toolType); err != nil {
		return err
	}
	switch toolType {
	case "function":
		err = decodeResponsesFunctionTool(body, request, policy, diagnostics)
	case "custom":
		err = decodeResponsesCustomTool(body, request, policy, diagnostics)
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "a Responses namespace can hold only function and custom tools", nil)
	}
	if err != nil {
		return err
	}
	tool := &request.Tools[len(request.Tools)-1]
	if strings.TrimSpace(tool.Name) == "" {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses namespace tool requires a name", nil)
	}
	tool.Namespace, tool.LocalName, tool.NamespaceDescription = namespace.Name, tool.Name, namespace.Description
	tool.Name = llmprotocol.FlattenToolName(namespace.Name, tool.Name)
	tool.Description = joinNamespaceDescription(namespace.Description, tool.Description)
	return nil
}

type responsesNamespaceEncodeWire struct {
	Type        string              `json:"type"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Tools       []responsesToolWire `json:"tools"`
}

// encodeResponsesToolList is the tools array of a Responses request: each
// tool as it is, or, where the request keeps namespaces, a tool flattened
// from one back under its namespace with its own name and description.
func encodeResponsesToolList(request llmprotocol.Request) []any {
	tools := make([]any, 0, len(request.Tools)+1)
	namespaces := make(map[string]int)
	for _, tool := range request.Tools {
		if !request.KeepToolNamespaces || tool.Namespace == "" {
			tools = append(tools, encodeResponsesTool(tool))
			continue
		}
		local := tool
		local.Name, local.Description = tool.LocalName, namespaceLocalDescription(tool)
		index, seen := namespaces[tool.Namespace]
		if !seen {
			index, namespaces[tool.Namespace] = len(tools), len(tools)
			tools = append(tools, &responsesNamespaceEncodeWire{Type: "namespace", Name: tool.Namespace, Description: tool.NamespaceDescription})
		}
		namespace := tools[index].(*responsesNamespaceEncodeWire)
		namespace.Tools = append(namespace.Tools, encodeResponsesTool(local))
	}
	return tools
}

// namespaceLocalDescription takes the namespace's description back off the
// front of a flattened tool's.
func namespaceLocalDescription(tool llmprotocol.Tool) string {
	switch tool.NamespaceDescription {
	case "":
		return tool.Description
	case tool.Description:
		return ""
	}
	return strings.TrimPrefix(tool.Description, tool.NamespaceDescription+"\n\n")
}

// restoreResponsesHistoryNamespaces names each history call by the namespace
// and name its client sent, where the request keeps namespaces.
func restoreResponsesHistoryNamespaces(request llmprotocol.Request, items []responsesItemWire) {
	if !request.KeepToolNamespaces {
		return
	}
	for index := range items {
		item := &items[index]
		if item.Type == "function_call" || item.Type == "custom_tool_call" {
			if namespace, name := request.ToolNameOrigins.Origin(item.Name); namespace != "" {
				item.Namespace, item.Name = namespace, name
			}
		}
	}
}

// A flat tool has no namespace left to describe it, so its description carries
// what the namespace told the model.
func joinNamespaceDescription(namespace, tool string) string {
	if namespace == "" {
		return tool
	}
	if tool == "" {
		return namespace
	}
	return namespace + "\n\n" + tool
}

// flattenResponsesHistoryCall gives a history call the flat name its tool is
// declared under, and claims that name so no different tool can share it.
func flattenResponsesHistoryCall(item *responsesItemWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if (item.Type != "function_call" && item.Type != "custom_tool_call") || strings.TrimSpace(item.Name) == "" {
		return nil
	}
	flat := item.Name
	if item.Namespace != "" {
		// Flattening shortens long parts, so bound them as the client sent them.
		if limit := policy.Limits.ToolNameBytes; limit > 0 && (len(item.Namespace) > limit || len(item.Name) > limit) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "tool_call_limit", "tool call namespace or name exceeds the configured limit", nil)
		}
		flat = llmprotocol.FlattenToolName(item.Namespace, item.Name)
		request.Generation++
	}
	if err := llmprotocol.ClaimToolName(request, flat, item.Namespace, item.Name); err != nil {
		return err
	}
	item.Name = flat
	return nil
}

// claimResponsesToolNames runs after the history calls are claimed, so a flat
// name cannot stand for two declared tools or for a declared tool and a
// different tool the history called. Duplicate plain tools share one origin and
// keep the validator's error.
func claimResponsesToolNames(request *llmprotocol.Request) error {
	for _, tool := range request.Tools {
		name := tool.Name
		if tool.Namespace != "" {
			name = tool.LocalName
		}
		if err := llmprotocol.ClaimToolName(request, tool.Name, tool.Namespace, name); err != nil {
			return err
		}
	}
	return nil
}

// projectToolCallNamespaces keeps the loss report for a Responses provider call
// that names a namespace on its way to a client format that has none.
func projectToolCallNamespaces(source, target llmprotocol.WireFormat, response *llmprotocol.Response) llmprotocol.Diagnostics {
	if target == llmprotocol.OpenAIResponsesV1 {
		return nil
	}
	var diagnostics llmprotocol.Diagnostics
	for _, item := range response.Output {
		for index := range item.Content {
			if item.Content[index].ToolCall == nil || item.Content[index].ToolCall.Namespace == "" {
				continue
			}
			call := *item.Content[index].ToolCall
			call.Namespace = ""
			item.Content[index].ToolCall = &call
			diagnostics = append(diagnostics, llmprotocol.Diagnostic{
				Source: source, Target: target, Field: "output.namespace",
				Action: llmprotocol.DiagnosticDropped, Reason: "the client format has no tool namespaces",
			})
		}
	}
	return diagnostics
}
