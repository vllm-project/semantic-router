package llmprotocol

import (
	"crypto/sha256"
	"encoding/hex"
	"maps"
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

// ToolNameOrigins maps each tool name a request declares or calls to the
// namespace and name its client sent; a plain name is its own origin. Flattening
// is not injective: mcp__a with b__c and mcp__a__b with c both become
// mcp__a__b__c, so a flat name may stand for only one origin.
type ToolNameOrigins map[string]toolNamespaceOrigin

// ClaimToolName records that the tool its client knows as namespace and name
// reaches backends as flat. It fails closed when a different tool already holds
// flat, because the model could not tell the two apart. An empty namespace
// claims a plain name.
func ClaimToolName(request *Request, flat, namespace, name string) error {
	if request.ToolNameOrigins == nil {
		request.ToolNameOrigins = make(ToolNameOrigins)
	}
	return request.ToolNameOrigins.claim(flat, toolNamespaceOrigin{namespace: namespace, name: name})
}

func (origins ToolNameOrigins) claim(flat string, origin toolNamespaceOrigin) error {
	if held, seen := origins[flat]; seen && held != origin {
		return NewError(ErrorInvalidRequest, "tool_namespace_collision",
			"namespaced tools flatten to the same name as another tool: "+flat, nil)
	}
	origins[flat] = origin
	return nil
}

// Flattened reports whether any name came from a namespace. Without one every
// name is its own origin, so a decoded request need not keep them.
func (origins ToolNameOrigins) Flattened() bool {
	for _, origin := range origins {
		if origin.namespace != "" {
			return true
		}
	}
	return false
}

// ClaimHistoryToolNames claims the names of retained history for the request it
// is prepended to, so a stored call cannot share a flat name with a different
// tool the request declares or calls.
func ClaimHistoryToolNames(request *Request, history Request) error {
	if request.ToolNameOrigins == nil && history.ToolNameOrigins == nil {
		return nil
	}
	merged := requestToolNameOrigins(*request)
	for flat, origin := range requestToolNameOrigins(history) {
		if err := merged.claim(flat, origin); err != nil {
			return err
		}
	}
	request.ToolNameOrigins = merged
	return nil
}

// requestToolNameOrigins copies a request's origins. A request that flattened
// nothing keeps none, so each of its names is its own origin.
func requestToolNameOrigins(request Request) ToolNameOrigins {
	origins := make(ToolNameOrigins, len(request.ToolNameOrigins))
	if request.ToolNameOrigins != nil {
		maps.Copy(origins, request.ToolNameOrigins)
		return origins
	}
	for _, tool := range request.Tools {
		origins[tool.Name] = toolNamespaceOrigin{name: tool.Name}
	}
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				origins[content.ToolCall.Name] = toolNamespaceOrigin{name: content.ToolCall.Name}
			}
		}
	}
	return origins
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
