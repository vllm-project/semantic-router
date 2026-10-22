package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const vendorExtensionReason = "provider vendor extension field is not part of the canonical response contract"

// providerVendorExtensionsAllowed reports whether the selected backend is a
// positively identified provider whose documented decorations may be stripped.
// The strip itself is provider-agnostic and reports every dropped path, so this
// allowlist only decides who is known to decorate responses: an unrecognized
// value keeps strict decoding instead of silently accepting decorations.
func providerVendorExtensionsAllowed(policy llmprotocol.Policy) bool {
	switch policy.ResponseVendor {
	case llmprotocol.ResponseVendorAzure, llmprotocol.ResponseVendorCloudflare,
		llmprotocol.ResponseVendorOpenAICompatGateway:
		return true
	}
	return false
}

// stripProviderVendorExtensions removes non-canonical fields and returns their
// sorted paths. It preserves the original bytes when no fields are removed.
func stripProviderVendorExtensions(body []byte, targetType reflect.Type) ([]byte, []string) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return body, nil
	}
	if trimmed[0] != '{' && trimmed[0] != '[' {
		return body, nil
	}
	dropped := map[string]struct{}{}
	rewritten, changed := stripVendorExtensionValue(trimmed, targetType, "", dropped)
	if !changed {
		return body, nil
	}
	paths := make([]string, 0, len(dropped))
	for path := range dropped {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return rewritten, paths
}

func stripVendorExtensionValue(
	raw json.RawMessage,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	targetType = dereferenceJSONType(targetType)
	if targetType == nil || len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) ||
		reflect.PointerTo(targetType).Implements(jsonUnmarshalerType) {
		return raw, false
	}
	switch targetType.Kind() {
	case reflect.Struct:
		return stripVendorExtensionObject(trimmed, targetType, path, dropped)
	case reflect.Slice, reflect.Array:
		return stripVendorExtensionArray(trimmed, targetType, path, dropped)
	}
	return raw, false
}

func stripVendorExtensionObject(
	raw json.RawMessage,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) (json.RawMessage, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return raw, false
	}
	fields := exactJSONStructFields(targetType)
	changed := false
	for name, value := range object {
		child := vendorExtensionPath(path, name)
		fieldType, found := fields[name]
		if !found {
			delete(object, name)
			dropped[child] = struct{}{}
			changed = true
			continue
		}
		if rewritten, childChanged := stripVendorExtensionValue(value, fieldType, child, dropped); childChanged {
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

func stripVendorExtensionArray(
	raw json.RawMessage,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) (json.RawMessage, bool) {
	if targetType.Elem() == reflect.TypeOf(json.RawMessage{}) {
		return raw, false
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return raw, false
	}
	changed := false
	elementPath := path + "[]"
	for index, element := range elements {
		if rewritten, elementChanged := stripVendorExtensionValue(
			element, targetType.Elem(), elementPath, dropped,
		); elementChanged {
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

// appendVendorExtensionDiagnostics records one dropped-field diagnostic per
// vendor decoration, bounded by the configured diagnostics limit.
func appendVendorExtensionDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source llmprotocol.WireFormat,
	fields []string,
) {
	for _, field := range fields {
		appendProviderFieldOmission(diagnostics, policy, source, field, vendorExtensionReason)
	}
}

// DiagnosticsDroppedVendorExtensions reports whether a decode dropped vendor
// response decorations. A body that produced these diagnostics is not canonical
// on the wire, so callers that persist a response must not persist those bytes:
// a strict reader would reject them later.
func DiagnosticsDroppedVendorExtensions(diagnostics llmprotocol.Diagnostics) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && diagnostic.Reason == vendorExtensionReason {
			return true
		}
	}
	return false
}

// gatewayUsageAliases maps the Anthropic-style usage counters that
// OpenAI-compatible aggregating gateways emit onto their canonical OpenAI
// names. The gateway vendor opts in to this translation; a canonical field
// already present takes precedence and the alias is dropped later as ordinary
// vendor metadata.
var gatewayUsageAliases = map[string]string{
	"input_tokens":  "prompt_tokens",
	"output_tokens": "completion_tokens",
}

// translateGatewayUsageCounters rewrites Anthropic-style usage counters into
// their canonical OpenAI names before the strict decode. Without it the strip
// would leave total_tokens standing alone against zero input/output totals and
// the usage consistency check would reject an otherwise sound gateway payload
// (issue #5046: input_tokens=12, output_tokens=178, total_tokens=190).
func translateGatewayUsageCounters(body []byte) []byte {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(body, &document); err != nil {
		return body
	}
	rawUsage, ok := document["usage"]
	if !ok {
		return body
	}
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(rawUsage, &usage); err != nil {
		return body
	}
	changed := false
	for alias, canonical := range gatewayUsageAliases {
		raw, ok := usage[alias]
		if !ok {
			continue
		}
		if _, exists := usage[canonical]; !exists {
			usage[canonical] = raw
			changed = true
		}
	}
	if !changed {
		return body
	}
	usageBytes, err := json.Marshal(usage)
	if err != nil {
		return body
	}
	document["usage"] = usageBytes
	rewritten, err := json.Marshal(document)
	if err != nil {
		return body
	}
	return rewritten
}
