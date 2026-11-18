package protocolcodec

import (
	"bytes"
	"encoding/json"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// captureRequestExtensions extracts the top-level request members an operator
// whitelisted through Policy.AllowedRequestExtensions and that the wire
// format itself does not declare, and returns them alongside the body without
// them. The stripped body is what the codec decodes, so every other unknown
// member keeps failing closed; the captured members ride the neutral request
// opaquely and same-format dispatch merges them back. The members' values are
// never interpreted or logged by the Router — their meaning belongs to the
// backend that declares them.
//
// Bodies that are not objects, or that do not parse, come back unchanged so
// the codec reports the original problem instead of a capture artifact.
func captureRequestExtensions(body []byte, policy llmprotocol.Policy) (map[string]json.RawMessage, []byte, error) {
	if len(policy.AllowedRequestExtensions) == 0 || len(body) == 0 || len(body) > policy.Limits.BodyBytes {
		return nil, body, nil
	}
	allowed := make(map[string]bool, len(policy.AllowedRequestExtensions))
	for _, name := range policy.AllowedRequestExtensions {
		if name != "" {
			allowed[name] = true
		}
	}
	if len(allowed) == 0 {
		return nil, body, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	token, err := decoder.Token()
	if err != nil {
		return nil, body, nil
	}
	if delim, ok := token.(json.Delim); !ok || delim != '{' {
		return nil, body, nil
	}
	var captured map[string]json.RawMessage
	type keptMember struct {
		key   string
		value json.RawMessage
	}
	var kept []keptMember
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return nil, body, nil
		}
		key, _ := keyToken.(string)
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, body, nil
		}
		if !allowed[key] {
			kept = append(kept, keptMember{key: key, value: append([]byte(nil), value...)})
			continue
		}
		if captured == nil {
			captured = make(map[string]json.RawMessage)
		}
		if _, duplicate := captured[key]; duplicate {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "duplicate_json_field",
				"JSON contains a duplicate field: "+key, nil)
		}
		captured[key] = json.RawMessage(append([]byte(nil), value...))
	}
	if captured == nil {
		return nil, body, nil
	}
	totalBytes := 0
	for _, value := range captured {
		totalBytes += len(value)
	}
	if totalBytes > policy.Limits.RequestExtensionBytes || len(captured) > policy.Limits.RequestExtensionMembers {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "extension_budget",
			"captured request extension members exceed the configured extension budget", nil)
	}
	// The stripped body only feeds the codec's IR extraction, so rebuilding
	// it from the kept members is safe; the original body stays intact for
	// source preservation and byte replay.
	var stripped []byte
	if len(kept) == 0 {
		stripped = []byte("{}")
	} else {
		buffer := bytes.Buffer{}
		buffer.WriteByte('{')
		for index, member := range kept {
			if index > 0 {
				buffer.WriteByte(',')
			}
			name, err := json.Marshal(member.key)
			if err != nil {
				return nil, body, nil
			}
			buffer.Write(name)
			buffer.WriteByte(':')
			buffer.Write(member.value)
		}
		buffer.WriteByte('}')
		stripped = buffer.Bytes()
	}
	return captured, stripped, nil
}

// mergeRequestExtensions appends the captured extension members to an encoded
// same-format body. A replayed body already carries them byte-identically and
// is returned untouched. Dispatch that crosses wire formats drops the members
// with a diagnostic instead of injecting them into a contract that does not
// declare them.
func mergeRequestExtensions(
	format llmprotocol.WireFormat,
	request llmprotocol.Request,
	envelope llmprotocol.Envelope,
	body []byte,
) ([]byte, llmprotocol.Diagnostics, error) {
	if len(request.ExtensionFields) == 0 || len(body) == 0 {
		return body, nil, nil
	}
	if len(envelope.Request) > 0 && bytes.Equal(body, envelope.Request) {
		return body, nil, nil // replayed bytes already carry the members
	}
	if request.ExtensionFieldsFormat != "" && request.ExtensionFieldsFormat != format {
		var diagnostics llmprotocol.Diagnostics
		for _, name := range sortedExtensionNames(request.ExtensionFields) {
			diagnostics = append(diagnostics, llmprotocol.Diagnostic{
				Source: request.ExtensionFieldsFormat, Target: format, Field: name,
				Action: llmprotocol.DiagnosticDropped,
				Reason: "request extension captured from another wire format",
			})
		}
		return body, diagnostics, nil
	}
	members := make([][]byte, 0, len(request.ExtensionFields))
	for _, name := range sortedExtensionNames(request.ExtensionFields) {
		encodedName, err := json.Marshal(name)
		if err != nil {
			return body, nil, nil // a name that cannot marshal is never injected
		}
		member := append(append(append([]byte(nil), encodedName...), ':'), request.ExtensionFields[name]...)
		members = append(members, member)
	}
	trimmed := bytes.TrimRight(body, " \t\r\n")
	if len(trimmed) < 2 || trimmed[len(trimmed)-1] != '}' {
		return body, nil, nil // not an object envelope; leave the body alone
	}
	separator := []byte(",")
	if trimmed[len(trimmed)-2] == '{' {
		separator = nil
	}
	merged := append([]byte(nil), trimmed[:len(trimmed)-1]...)
	merged = append(merged, separator...)
	merged = append(merged, bytes.Join(members, []byte(","))...)
	merged = append(merged, '}')
	return merged, nil, nil
}

// sortedExtensionNames keeps merged and dropped output deterministic.
func sortedExtensionNames(members map[string]json.RawMessage) []string {
	names := make([]string, 0, len(members))
	for name := range members {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
