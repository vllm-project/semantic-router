package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var (
	errDuplicateJSONField = errors.New("duplicate JSON field")
	errInvalidJSONUnicode = errors.New("invalid JSON Unicode escape")
	errTrailingJSON       = errors.New("trailing JSON document")
)

func decodeWire(body []byte, target any, policy llmprotocol.Policy) error {
	if len(body) == 0 || len(body) > policy.Limits.BodyBytes {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "body_limit", "request body is empty or exceeds the configured limit", nil)
	}
	if !utf8.Valid(body) {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_utf8", "request JSON is not valid UTF-8", nil)
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_unicode", "request JSON contains an unpaired Unicode surrogate", err)
	}
	if !hasJSONObjectEnvelope(body) {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request body must be a JSON object", nil)
	}
	if err := validateNoDuplicateKeys(body, policy.Limits.JSONDepth); err != nil {
		if errors.Is(err, errDuplicateJSONField) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "duplicate_json_field", "JSON contains a duplicate field", err)
		}
		if errors.Is(err, errTrailingJSON) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "trailing_json", "request body contains trailing JSON", err)
		}
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON is invalid", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if rejectUnknownFields(body, policy) {
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(target); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON is invalid", err)
	}
	if err := requireEOF(decoder); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "trailing_json", "request body contains trailing JSON", err)
	}
	return nil
}

func decodeProviderWire(body []byte, target any, policy llmprotocol.Policy) error {
	if len(body) == 0 || len(body) > policy.Limits.BodyBytes {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit", "upstream response is empty or too large", nil)
	}
	if !utf8.Valid(body) {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8", "upstream response JSON is not valid UTF-8", nil)
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_unicode", "upstream response JSON contains an unpaired Unicode surrogate", err)
	}
	if !hasJSONObjectEnvelope(body) {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response body must be a JSON object", nil)
	}
	if err := validateNoDuplicateKeys(body, policy.Limits.JSONDepth); err != nil {
		if errors.Is(err, errDuplicateJSONField) {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_duplicate_json_field", "upstream response contains a duplicate field", err)
		}
		if errors.Is(err, errTrailingJSON) {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_trailing_json", "upstream response contains trailing JSON", err)
		}
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON is invalid", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if rejectUnknownFields(body, policy) {
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(target); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON is invalid", err)
	}
	if err := requireEOF(decoder); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_trailing_json", "upstream response contains trailing JSON", err)
	}
	return nil
}

func hasJSONObjectEnvelope(body []byte) bool {
	trimmed := bytes.TrimSpace(body)
	return len(trimmed) > 0 && trimmed[0] == '{'
}

// encoding/json replaces unpaired UTF-16 surrogates with U+FFFD. Translation
// must reject that malformed wire input instead of silently changing model
// text, identifiers, tool arguments, or signatures.
func validateJSONUnicodeEscapes(body []byte) error {
	insideString := false
	for index := 0; index < len(body); index++ {
		switch body[index] {
		case '"':
			insideString = !insideString
		case '\\':
			if !insideString || index+1 >= len(body) {
				continue
			}
			if body[index+1] != 'u' {
				index++
				continue
			}
			value, ok := decodeHexQuad(body, index+2)
			if !ok {
				continue
			}
			if value >= 0xdc00 && value <= 0xdfff {
				return fmt.Errorf("%w: lone low surrogate", errInvalidJSONUnicode)
			}
			if value < 0xd800 || value > 0xdbff {
				index += 5
				continue
			}
			if index+11 >= len(body) || body[index+6] != '\\' || body[index+7] != 'u' {
				return fmt.Errorf("%w: high surrogate is not followed by a low surrogate", errInvalidJSONUnicode)
			}
			low, validLow := decodeHexQuad(body, index+8)
			if !validLow || low < 0xdc00 || low > 0xdfff {
				return fmt.Errorf("%w: high surrogate is not followed by a low surrogate", errInvalidJSONUnicode)
			}
			index += 11
		}
	}
	return nil
}

func decodeHexQuad(body []byte, start int) (uint16, bool) {
	if start < 0 || start+4 > len(body) {
		return 0, false
	}
	var value uint16
	for _, character := range body[start : start+4] {
		value <<= 4
		switch {
		case character >= '0' && character <= '9':
			value += uint16(character - '0')
		case character >= 'a' && character <= 'f':
			value += uint16(character-'a') + 10
		case character >= 'A' && character <= 'F':
			value += uint16(character-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}

// decodeProviderEventType validates the complete JSON envelope before reading
// only its discriminator. Stream codecs use this first so a known but
// unsupported union variant fails with a typed capability error rather than an
// unrelated unknown-field error from a supported-event wire struct.
func decodeProviderEventType(body []byte, fallback string, policy llmprotocol.Policy) (string, error) {
	var fields map[string]json.RawMessage
	if err := decodeProviderWire(body, &fields, policy); err != nil {
		return "", err
	}
	eventType := fallback
	if raw := fields["type"]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &eventType); err != nil || eventType == "" {
			return "", llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"invalid_upstream_event_type",
				"upstream stream event type is invalid",
				err,
			)
		}
	}
	if eventType == "" {
		return "", llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"missing_upstream_event_type",
			"upstream stream event type is missing",
			nil,
		)
	}
	if fallback != "" && fallback != eventType {
		return "", llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_event_type_mismatch",
			"SSE event name does not match its JSON event type",
			nil,
		)
	}
	return eventType, nil
}

func rejectUnknownFields(body []byte, policy llmprotocol.Policy) bool {
	if policy.UnknownFields == llmprotocol.UnknownReject {
		return true
	}
	return policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat &&
		(policy.SourcePreservation != llmprotocol.SourceBoundedSameFormat ||
			policy.Limits.SourceEnvelopeBytes <= 0 || len(body) > policy.Limits.SourceEnvelopeBytes)
}

func requireEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("additional JSON document")
		}
		return err
	}
	return nil
}

func validateNoDuplicateKeys(body []byte, maximumDepth int) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := consumeJSONValue(decoder, 0, maximumDepth); err != nil {
		return err
	}
	if err := requireEOF(decoder); err != nil {
		return fmt.Errorf("%w: %w", errTrailingJSON, err)
	}
	return nil
}

// isJSONObject accepts exactly one top-level JSON object. Streamed tool
// arguments use the same configurable nesting and duplicate-key limits as the
// rest of the wire contract instead of relying on json.Valid's unbounded,
// shape-agnostic check.
func isJSONObject(body []byte, maximumDepth int) bool {
	return llmprotocol.ValidateJSONObject(body, maximumDepth) == nil
}

func consumeJSONValue(decoder *json.Decoder, depth, maximumDepth int) error {
	if maximumDepth <= 0 || depth > maximumDepth {
		return fmt.Errorf("JSON nesting exceeds the configured limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		return consumeJSONObject(decoder, depth, maximumDepth)
	case '[':
		return consumeJSONArray(decoder, depth, maximumDepth)
	default:
		return fmt.Errorf("unexpected delimiter %q", delimiter)
	}
}

func consumeJSONObject(decoder *json.Decoder, depth, maximumDepth int) error {
	seen := make(map[string]struct{})
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := keyToken.(string)
		if !ok {
			return fmt.Errorf("object key is not a string")
		}
		// encoding/json matches struct fields case-insensitively after trying an
		// exact match. Reject case-folded collisions before decoding so a second
		// spelling cannot silently replace an authenticated or routed value.
		canonicalKey := strings.ToLower(key)
		if _, duplicate := seen[canonicalKey]; duplicate {
			return fmt.Errorf("%w %q", errDuplicateJSONField, key)
		}
		seen[canonicalKey] = struct{}{}
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return fmt.Errorf("unterminated object")
	}
	return nil
}

func consumeJSONArray(decoder *json.Decoder, depth, maximumDepth int) error {
	for decoder.More() {
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim(']') {
		return fmt.Errorf("unterminated array")
	}
	return nil
}

func requestEnvelope(format llmprotocol.WireFormat, body []byte, generation uint64, policy llmprotocol.Policy) llmprotocol.Envelope {
	envelope := llmprotocol.Envelope{Format: format, Generation: generation}
	if policy.SourcePreservation == llmprotocol.SourceBoundedSameFormat &&
		policy.Limits.SourceEnvelopeBytes > 0 && len(body) <= policy.Limits.SourceEnvelopeBytes {
		envelope.Request = append([]byte(nil), body...)
	}
	return envelope
}

func responseEnvelope(format llmprotocol.WireFormat, body []byte, generation uint64, stop string, policy llmprotocol.Policy) llmprotocol.Envelope {
	envelope := llmprotocol.Envelope{Format: format, Generation: generation, SourceStop: stop}
	if policy.SourcePreservation == llmprotocol.SourceBoundedSameFormat &&
		policy.Limits.SourceEnvelopeBytes > 0 && len(body) <= policy.Limits.SourceEnvelopeBytes {
		envelope.Response = append([]byte(nil), body...)
	}
	return envelope
}

func appendLossy(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source, target llmprotocol.WireFormat, field, reason string) error {
	if policy.LossyFeatures == llmprotocol.LossyReject {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "lossy_translation", "translation would lose "+field, nil)
	}
	if len(*diagnostics) < policy.Limits.Diagnostics {
		*diagnostics = append(*diagnostics, llmprotocol.Diagnostic{
			Source: source, Target: target, Field: field,
			Action: llmprotocol.DiagnosticApproximated, Reason: reason,
		})
	}
	return nil
}

func rejectUnsupportedRequestField(field string, value json.RawMessage) error {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"unsupported_"+strings.ReplaceAll(field, ".", "_"),
		field+" is not supported by the protocol-neutral request contract",
		nil,
	)
}

func rejectUnsupportedRequestFields(fields map[string]json.RawMessage) error {
	names := make([]string, 0, len(fields))
	for name, value := range fields {
		trimmed := bytes.TrimSpace(value)
		if len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return nil
	}
	return rejectUnsupportedRequestField(names[0], fields[names[0]])
}

// appendAccountingOmission records a representation-only omission. Semantic
// usage remains available to settlement even when a client format has no field
// for a detailed accounting bucket, so this never converts a successful
// backend response into a protocol failure.
func appendAccountingOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source, target llmprotocol.WireFormat, field, reason string) {
	*diagnostics = appendDiagnostics(*diagnostics, llmprotocol.Diagnostics{{
		Source: source, Target: target, Field: field,
		Action: llmprotocol.DiagnosticDropped, Reason: reason,
	}}, policy.Limits.Diagnostics)
}

func appendProviderFieldOmission(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source llmprotocol.WireFormat,
	field,
	reason string,
) {
	*diagnostics = appendDiagnostics(*diagnostics, llmprotocol.Diagnostics{{
		Source: source, Field: field,
		Action: llmprotocol.DiagnosticDropped, Reason: reason,
	}}, policy.Limits.Diagnostics)
}

func appendProviderFieldOmissions(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source llmprotocol.WireFormat,
	fields map[string]bool,
	reason string,
) {
	names := make([]string, 0, len(fields))
	for name, present := range fields {
		if present {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		appendProviderFieldOmission(diagnostics, policy, source, name, reason)
	}
}

func canonicalRole(value string) (llmprotocol.Role, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "system":
		return llmprotocol.RoleSystem, nil
	case "developer":
		return llmprotocol.RoleDeveloper, nil
	case "user":
		return llmprotocol.RoleUser, nil
	case "assistant":
		return llmprotocol.RoleAssistant, nil
	case "tool":
		return llmprotocol.RoleTool, nil
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_role", "message role is invalid", nil)
	}
}

func wireRole(role llmprotocol.Role) (string, error) {
	switch role {
	case llmprotocol.RoleSystem, llmprotocol.RoleDeveloper, llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleTool:
		return string(role), nil
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_role", "message role is invalid", nil)
	}
}

func authoritative(value int64) llmprotocol.TokenCount {
	return llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageAuthoritative}
}

func unknownCount() llmprotocol.TokenCount {
	return llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
}

func tokenValue(value llmprotocol.TokenCount) int64 {
	if value.Value == nil {
		return 0
	}
	return *value.Value
}

func decodeDataURL(raw string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(raw, "data:") {
		return "", "", false
	}
	header, data, found := strings.Cut(strings.TrimPrefix(raw, "data:"), ",")
	if !found || data == "" || !strings.HasSuffix(strings.ToLower(header), ";base64") {
		return "", "", false
	}
	mediaType = header[:len(header)-len(";base64")]
	if strings.TrimSpace(mediaType) == "" {
		return "", "", false
	}
	return mediaType, data, true
}

func marshalWire(value any) ([]byte, error) {
	body, err := json.Marshal(value)
	if err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "encode_wire", "wire response could not be encoded", err)
	}
	return body, nil
}
