package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vendorExtensionReason is recorded on every dropped field so a diagnostic
// reader can tell a vendor decoration apart from a field the router chose not
// to carry across formats.
const vendorExtensionReason = "provider vendor extension field is not part of the canonical response contract"

// providerVendorExtensionsAllowed reports whether the backend that produced
// this response was identified as a vendor known to decorate its responses.
//
// The allowance is deliberately per-backend rather than per-field-name. Azure
// has emitted content_filter_results for years and ships new decorations across
// API versions, so a fixed name list would fail the first time they add one -
// the exact failure #3496 reports. Scoping by resolved dialect keeps every
// other backend under the strict canonical contract.
func providerVendorExtensionsAllowed(policy llmprotocol.Policy) bool {
	switch policy.ResponseVendor {
	case llmprotocol.VendorAzure:
		return true
	default:
		return false
	}
}

// stripProviderVendorExtensions removes the fields a vendor-identified backend
// added outside the canonical schema, returning the rewritten body and the
// sorted paths it removed. Nothing is dropped silently: every path is reported
// so the caller can raise a diagnostic for it.
//
// When nothing matches, the original slice is returned so the common path stays
// allocation-free. Object key order is not preserved by the rewrite, which is
// why this never runs on the byte-identical source-preservation path -
// rejectUnknownFields is false there.
func stripProviderVendorExtensions(body []byte, targetType reflect.Type) ([]byte, []string) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return body, nil
	}
	unknown := map[string]struct{}{}
	collectUnknownJSONFieldPaths(trimmed, dereferenceJSONType(targetType), "", unknown)
	if len(unknown) == 0 {
		return body, nil
	}
	rewritten, changed := stripVendorExtensionValue(trimmed, "", unknown)
	if !changed {
		return body, nil
	}
	paths := make([]string, 0, len(unknown))
	for path := range unknown {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return rewritten, paths
}

// collectUnknownJSONFieldPaths mirrors validateExactJSONValue, recording the
// normalized path of every field the canonical schema does not declare instead
// of failing on the first one. "[]" denotes an array element, so one path
// covers every element of a repeated field.
func collectUnknownJSONFieldPaths(body []byte, targetType reflect.Type, path string, unknown map[string]struct{}) {
	if targetType == nil || bytes.Equal(bytes.TrimSpace(body), []byte("null")) ||
		reflect.PointerTo(targetType).Implements(jsonUnmarshalerType) {
		return
	}
	switch targetType.Kind() {
	case reflect.Struct:
		collectUnknownJSONObjectPaths(body, targetType, path, unknown)
	case reflect.Slice, reflect.Array:
		collectUnknownJSONArrayPaths(body, targetType, path, unknown)
	}
}

func collectUnknownJSONObjectPaths(body []byte, targetType reflect.Type, path string, unknown map[string]struct{}) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return
	}
	fields := exactJSONStructFields(targetType)
	for name, value := range object {
		child := vendorExtensionPath(path, name)
		fieldType, found := fields[name]
		if !found {
			unknown[child] = struct{}{}
			continue
		}
		collectUnknownJSONFieldPaths(value, dereferenceJSONType(fieldType), child, unknown)
	}
}

func collectUnknownJSONArrayPaths(body []byte, targetType reflect.Type, path string, unknown map[string]struct{}) {
	if targetType.Elem() == reflect.TypeOf(json.RawMessage{}) {
		return
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(body, &elements); err != nil {
		return
	}
	elementType := dereferenceJSONType(targetType.Elem())
	elementPath := path + "[]"
	for _, element := range elements {
		collectUnknownJSONFieldPaths(element, elementType, elementPath, unknown)
	}
}

func stripVendorExtensionValue(raw json.RawMessage, path string, drop map[string]struct{}) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return raw, false
	}
	switch trimmed[0] {
	case '{':
		return stripVendorExtensionObject(trimmed, path, drop)
	case '[':
		return stripVendorExtensionArray(trimmed, path, drop)
	}
	return raw, false
}

func stripVendorExtensionObject(raw json.RawMessage, path string, drop map[string]struct{}) (json.RawMessage, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		// Malformed JSON is not this function's error to report; leave the body
		// untouched so the decoder produces the canonical failure.
		return raw, false
	}
	changed := false
	for name, value := range object {
		child := vendorExtensionPath(path, name)
		if _, dropped := drop[child]; dropped {
			delete(object, name)
			changed = true
			continue
		}
		if rewritten, childChanged := stripVendorExtensionValue(value, child, drop); childChanged {
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

func stripVendorExtensionArray(raw json.RawMessage, path string, drop map[string]struct{}) (json.RawMessage, bool) {
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return raw, false
	}
	changed := false
	elementPath := path + "[]"
	for index, element := range elements {
		if rewritten, elementChanged := stripVendorExtensionValue(element, elementPath, drop); elementChanged {
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
