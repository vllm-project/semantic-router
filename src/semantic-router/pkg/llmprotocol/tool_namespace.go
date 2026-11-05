package llmprotocol

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// Chat and Messages have no tool grouping, and their name limits are narrower
// than a Responses namespace allows, so ingress flattens each namespaced tool
// into a portable name no backend can reject.
const (
	toolNamespaceSeparator = "__"
	flatToolNameBytes      = 64
	flatToolLocalBytes     = 32
	flatToolDigestChars    = 12
)

// FlattenToolName derives the backend name of a namespaced tool from its
// namespace and name alone, so a history call flattens to the declared name even
// after the tool has left the request. The name is not parsed back: separators
// occur inside namespaces such as mcp__server, so ToolNamespaces restores it.
func FlattenToolName(namespace, name string) string {
	joined := namespace + toolNamespaceSeparator + name
	if len(joined) <= flatToolNameBytes && isPortableToolName(joined) {
		return joined
	}
	digest := sha256.Sum256([]byte(namespace + "\x00" + name))
	suffix := hex.EncodeToString(digest[:])[:flatToolDigestChars] + toolNamespaceSeparator +
		truncateToolNamePart(portableToolName(name), flatToolLocalBytes)
	prefix := truncateToolNamePart(portableToolName(namespace), flatToolNameBytes-len(suffix)-1)
	return prefix + "_" + suffix
}

func isPortableToolName(name string) bool {
	return strings.IndexFunc(name, func(char rune) bool { return !isPortableToolNameChar(char) }) < 0
}

func isPortableToolNameChar(char rune) bool {
	return char == '_' || char == '-' || char >= '0' && char <= '9' ||
		char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z'
}

func portableToolName(name string) string {
	return strings.Map(func(char rune) rune {
		if isPortableToolNameChar(char) {
			return char
		}
		return '_'
	}, name)
}

func truncateToolNamePart(part string, limit int) string {
	if len(part) > limit {
		return part[:limit]
	}
	return part
}

type toolNamespaceOrigin struct {
	namespace string
	name      string
}

// ToolNamespaces maps the flat names a request declared back to the namespace
// and name its client knows.
type ToolNamespaces map[string]toolNamespaceOrigin

// RequestToolNamespaces returns nil when no tool came from a namespace.
func RequestToolNamespaces(tools []Tool) ToolNamespaces {
	var namespaces ToolNamespaces
	for _, tool := range tools {
		if tool.Namespace == "" {
			continue
		}
		if namespaces == nil {
			namespaces = make(ToolNamespaces)
		}
		namespaces[tool.Name] = toolNamespaceOrigin{namespace: tool.Namespace, name: tool.LocalName}
	}
	return namespaces
}

// RestoreCall returns a namespaced copy of a call to a flattened tool. A call to
// any other tool is returned as it is.
func (namespaces ToolNamespaces) RestoreCall(call *ToolCall) *ToolCall {
	if call == nil {
		return nil
	}
	origin, flattened := namespaces[call.Name]
	if !flattened {
		return call
	}
	restored := *call
	restored.Namespace, restored.Name = origin.namespace, origin.name
	return &restored
}

// RestoreResponse restores every call in a response's output and alternatives.
func (namespaces ToolNamespaces) RestoreResponse(response *Response) {
	if response == nil || len(namespaces) == 0 {
		return
	}
	namespaces.restoreOutput(response.Output)
	for _, alternative := range response.Alternatives {
		namespaces.restoreOutput(alternative)
	}
}

func (namespaces ToolNamespaces) restoreOutput(items []OutputItem) {
	for _, item := range items {
		for index := range item.Content {
			item.Content[index].ToolCall = namespaces.RestoreCall(item.Content[index].ToolCall)
		}
	}
}
