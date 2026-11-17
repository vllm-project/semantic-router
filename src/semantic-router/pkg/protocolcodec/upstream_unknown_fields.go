package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const unknownUpstreamFieldReason = "upstream field is not part of the wire contract; a same-format reply forwards it unchanged"

// maxUnknownFieldPathBytes bounds one reported path. Member names come from
// the upstream, and every reported path reaches the warning header and the
// debug log.
const maxUnknownFieldPathBytes = 256

// dropUnknownUpstreamFields reports whether upstream decoding drops the
// members a wire contract does not declare, reporting each one, instead of
// failing. Every backend decodes this way: an unknown member never fails a
// reply, a stream frame or an error envelope, and never reaches the neutral
// response or accounting. Only conformance tests reject unknown members.
func dropUnknownUpstreamFields(policy llmprotocol.Policy) bool {
	return !policy.RejectUnknownUpstreamFields
}

// stripUnknownUpstreamFields removes the members targetType does not declare
// and returns their sorted paths. It preserves the original bytes when no
// member is removed.
func stripUnknownUpstreamFields(body []byte, targetType reflect.Type) ([]byte, []string) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return body, nil
	}
	if trimmed[0] != '{' && trimmed[0] != '[' {
		return body, nil
	}
	dropped := map[string]struct{}{}
	rewritten, changed := stripUnknownFieldValue(trimmed, targetType, "", dropped)
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

func stripUnknownFieldValue(
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
		return stripUnknownFieldObject(trimmed, targetType, path, dropped)
	case reflect.Slice, reflect.Array:
		return stripUnknownFieldArray(trimmed, targetType, path, dropped)
	}
	return raw, false
}

func stripUnknownFieldObject(
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
		child := unknownFieldPath(path, name)
		fieldType, found := fields[name]
		if !found {
			delete(object, name)
			dropped[child] = struct{}{}
			changed = true
			continue
		}
		if rewritten, childChanged := stripUnknownFieldValue(value, fieldType.typeOf, child, dropped); childChanged {
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

func stripUnknownFieldArray(
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
		if rewritten, elementChanged := stripUnknownFieldValue(
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

func unknownFieldPath(parent, name string) string {
	path := name
	if parent != "" {
		path = parent + "." + name
	}
	if len(path) > maxUnknownFieldPathBytes {
		path = string(bytes.ToValidUTF8([]byte(path[:maxUnknownFieldPathBytes]), nil)) + "..."
	}
	return path
}

// rejectRouterExtensionFields refuses a Chat body whose top level carries one
// of the Router's own extension names. They are known names, not unknown
// members: dropping them would still let a same-format replay forward the
// upstream's value as Router evidence.
func rejectRouterExtensionFields(unknownFields []string) error {
	for _, field := range unknownFields {
		if llmprotocol.IsRouterExtensionField(field) {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json",
				"upstream response JSON contains a non-canonical field",
				fmt.Errorf("upstream member %q is a Router extension field", field),
			)
		}
	}
	return nil
}

// appendUnknownFieldDiagnostics records one dropped-field diagnostic per
// unknown member, bounded by the configured diagnostics limit. The extproc
// boundary turns each one into llm_translation_lossy_total and a debug event.
func appendUnknownFieldDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source llmprotocol.WireFormat,
	fields []string,
) {
	for _, field := range fields {
		appendProviderFieldOmission(diagnostics, policy, source, field, unknownUpstreamFieldReason)
	}
}

// DiagnosticsDroppedUnknownFields reports whether a decode dropped upstream
// members its wire contract does not declare. Such a body is not canonical on
// the wire, so a caller that persists a response for other clients must store
// the canonical encoding instead of those bytes.
func DiagnosticsDroppedUnknownFields(diagnostics llmprotocol.Diagnostics) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && diagnostic.Reason == unknownUpstreamFieldReason {
			return true
		}
	}
	return false
}

// chatUsageCounterAliases maps the Anthropic-style usage counters that
// OpenAI-compatible relays emit in Chat usage onto their Chat names. A Chat
// name already present takes precedence, and the alias is then dropped as an
// unknown member.
var chatUsageCounterAliases = map[string]string{
	"input_tokens":  "prompt_tokens",
	"output_tokens": "completion_tokens",
}

// aliasChatUsageCounters rewrites Anthropic-style usage counters in a Chat
// body into their Chat names before the decode. Relays that put a Claude
// channel behind an OpenAI face keep those names (issue #5046:
// input_tokens=12, output_tokens=178, total_tokens=190); dropping them as
// unknown would leave total_tokens alone against zero prompt and completion
// counts, and the usage totals check would reject a sound reply.
func aliasChatUsageCounters(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"input_tokens"`)) && !bytes.Contains(body, []byte(`"output_tokens"`)) {
		return body
	}
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
	for alias, canonical := range chatUsageCounterAliases {
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
