package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const vendorExtensionReason = "provider vendor extension field is not part of the canonical response contract"

func providerVendorExtensionsAllowed(policy llmprotocol.Policy) bool {
	return policy.ResponseVendor == llmprotocol.ResponseVendorAzure
}

// stripProviderVendorExtensions removes non-canonical fields and returns their
// sorted paths. It preserves the original bytes when no fields are removed.
func stripProviderVendorExtensions(body []byte, targetType reflect.Type) ([]byte, []string) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
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
