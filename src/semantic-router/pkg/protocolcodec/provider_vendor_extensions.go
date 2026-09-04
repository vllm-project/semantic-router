package protocolcodec

import (
	"bytes"
	"encoding/json"
	"sort"
)

// providerVendorExtensionFields lists provider response fields that are known,
// documented vendor decorations rather than canonical protocol fields. They are
// keyed by their normalized path from the response envelope, where "[]" denotes
// an array element, so a name is only ever ignored where that provider actually
// emits it.
//
// Strict rejection of unknown provider fields is deliberate: a field the router
// does not understand must never disappear silently on the way to the client.
// These entries are the narrow exception. Each one is inert telemetry or
// moderation metadata that carries no routing, billing, or content semantics,
// so dropping it cannot change what the client sees. Anything not listed here
// still fails decode with invalid_upstream_json.
var providerVendorExtensionFields = map[string]string{
	// Azure OpenAI and Azure AI Foundry decorate every response.
	"prompt_filter_results":            "azure",
	"routing":                          "azure",
	"choices[].content_filter_results": "azure",
	"usage.latency_checkpoint":         "azure",
}

// stripProviderVendorExtensions removes known vendor extension fields from a
// provider response body before strict canonical validation runs. It returns
// the rewritten body and the sorted paths it removed; when nothing matches it
// returns the original slice so the common path stays allocation-free.
//
// The rewrite only ever deletes allowlisted keys. Values are carried through as
// raw JSON, so numbers, escapes, and nesting survive byte-for-byte. Object key
// order is not preserved, which is why this never runs on the byte-identical
// source-preservation path: rejectUnknownFields is false there.
func stripProviderVendorExtensions(body []byte) ([]byte, []string) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return body, nil
	}
	removed := map[string]struct{}{}
	rewritten, changed := stripVendorExtensionValue(trimmed, "", removed)
	if !changed {
		return body, nil
	}
	paths := make([]string, 0, len(removed))
	for path := range removed {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return rewritten, paths
}

func stripVendorExtensionValue(raw json.RawMessage, path string, removed map[string]struct{}) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return raw, false
	}
	switch trimmed[0] {
	case '{':
		return stripVendorExtensionObject(trimmed, path, removed)
	case '[':
		return stripVendorExtensionArray(trimmed, path, removed)
	}
	return raw, false
}

func stripVendorExtensionObject(raw json.RawMessage, path string, removed map[string]struct{}) (json.RawMessage, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		// Malformed JSON is not this function's error to report; leave the body
		// untouched so the decoder produces the canonical failure.
		return raw, false
	}
	changed := false
	for name, value := range object {
		child := vendorExtensionPath(path, name)
		if _, vendor := providerVendorExtensionFields[child]; vendor {
			delete(object, name)
			removed[child] = struct{}{}
			changed = true
			continue
		}
		if rewritten, childChanged := stripVendorExtensionValue(value, child, removed); childChanged {
			object[name] = rewritten
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	rewritten, err := json.Marshal(object)
	if err != nil {
		return raw, false
	}
	return rewritten, true
}

func stripVendorExtensionArray(raw json.RawMessage, path string, removed map[string]struct{}) (json.RawMessage, bool) {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return raw, false
	}
	changed := false
	elementPath := path + "[]"
	for index, element := range elements {
		if rewritten, elementChanged := stripVendorExtensionValue(element, elementPath, removed); elementChanged {
			elements[index] = rewritten
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	rewritten, err := json.Marshal(elements)
	if err != nil {
		return raw, false
	}
	return rewritten, true
}

func vendorExtensionPath(parent, name string) string {
	if parent == "" {
		return name
	}
	return parent + "." + name
}
