package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
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
	return decodeWireJSON(body, target, policy, true)
}

// decodeWireValue validates and decodes a nested client JSON value. Unlike a
// request envelope, a nested value may be an array or scalar. It still receives
// the same size, UTF-8, duplicate-key, depth, unknown-field, and trailing-data
// enforcement as the top-level document.
func decodeWireValue(body []byte, target any, policy llmprotocol.Policy) error {
	return decodeWireJSON(body, target, policy, false)
}

func decodeWireJSON(body []byte, target any, policy llmprotocol.Policy, requireObject bool) error {
	if err := validateClientJSONDocument(body, policy, requireObject); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if rejectUnknownFields(body, policy) {
		if err := validateExactJSONFieldNames(body, reflect.TypeOf(target)); err != nil {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON contains a non-canonical field", err)
		}
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

func validateClientJSONDocument(body []byte, policy llmprotocol.Policy, requireObject bool) error {
	if len(body) == 0 || len(body) > policy.Limits.BodyBytes {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "body_limit", "request body is empty or exceeds the configured limit", nil)
	}
	if !utf8.Valid(body) {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_utf8", "request JSON is not valid UTF-8", nil)
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_unicode", "request JSON contains an unpaired Unicode surrogate", err)
	}
	if requireObject && !hasJSONObjectEnvelope(body) {
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
	return nil
}

func decodeProviderWire(body []byte, target any, policy llmprotocol.Policy) error {
	return decodeProviderJSON(body, target, policy, true)
}

// decodeProviderValue is the upstream counterpart to decodeWireValue. Provider
// response envelopes remain object-only, while their typed nested arrays and
// scalars use this path without weakening any other JSON validation.
func decodeProviderValue(body []byte, target any, policy llmprotocol.Policy) error {
	return decodeProviderJSON(body, target, policy, false)
}

func decodeProviderJSON(body []byte, target any, policy llmprotocol.Policy, requireObject bool) error {
	if err := validateProviderJSONDocument(body, policy, requireObject); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if rejectUnknownFields(body, policy) {
		if err := validateExactJSONFieldNames(body, reflect.TypeOf(target)); err != nil {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON contains a non-canonical field", err)
		}
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

func validateProviderJSONDocument(body []byte, policy llmprotocol.Policy, requireObject bool) error {
	if len(body) == 0 || len(body) > policy.Limits.BodyBytes {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit", "upstream response is empty or too large", nil)
	}
	if !utf8.Valid(body) {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8", "upstream response JSON is not valid UTF-8", nil)
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_unicode", "upstream response JSON contains an unpaired Unicode surrogate", err)
	}
	if requireObject && !hasJSONObjectEnvelope(body) {
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
	return nil
}

func hasJSONObjectEnvelope(body []byte) bool {
	trimmed := bytes.TrimSpace(body)
	return len(trimmed) > 0 && trimmed[0] == '{'
}

var jsonUnmarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()

// encoding/json matches struct fields case-insensitively. JSON object member
// names are case-sensitive, and every supported wire schema publishes exact
// names, so accepting variants such as "messAges" would let malformed input
// bypass unknown-field rejection. Validate the concrete wire shape before the
// standard decoder runs. Dynamic maps and RawMessage fields remain open by
// design and are validated when a codec later decodes them into a typed union.
func validateExactJSONFieldNames(body []byte, targetType reflect.Type) error {
	return validateExactJSONValue(body, dereferenceJSONType(targetType))
}

func validateExactJSONValue(body []byte, targetType reflect.Type) error {
	if targetType == nil || bytes.Equal(bytes.TrimSpace(body), []byte("null")) ||
		reflect.PointerTo(targetType).Implements(jsonUnmarshalerType) {
		return nil
	}
	switch targetType.Kind() {
	case reflect.Struct:
		return validateExactJSONObject(body, targetType)
	case reflect.Slice, reflect.Array:
		return validateExactJSONArray(body, targetType)
	case reflect.Map, reflect.Interface:
		return nil
	}
	return nil
}

func validateExactJSONObject(body []byte, targetType reflect.Type) error {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return err
	}
	fields := exactJSONStructFields(targetType)
	for name, value := range object {
		fieldType, found := fields[name]
		if !found {
			return fmt.Errorf("unknown field %q", name)
		}
		if err := validateExactJSONValue(value, dereferenceJSONType(fieldType)); err != nil {
			return fmt.Errorf("field %q: %w", name, err)
		}
	}
	return nil
}

func validateExactJSONArray(body []byte, targetType reflect.Type) error {
	if targetType.Elem() == reflect.TypeOf(json.RawMessage{}) {
		return nil
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(body, &elements); err != nil {
		return err
	}
	for index, element := range elements {
		if err := validateExactJSONValue(element, dereferenceJSONType(targetType.Elem())); err != nil {
			return fmt.Errorf("element %d: %w", index, err)
		}
	}
	return nil
}

func exactJSONStructFields(targetType reflect.Type) map[string]reflect.Type {
	fields := make(map[string]reflect.Type)
	for index := 0; index < targetType.NumField(); index++ {
		field := targetType.Field(index)
		if field.PkgPath != "" {
			continue
		}
		tag := field.Tag.Get("json")
		name := strings.Split(tag, ",")[0]
		if name == "-" {
			continue
		}
		if field.Anonymous && name == "" {
			embedded := dereferenceJSONType(field.Type)
			if embedded.Kind() == reflect.Struct {
				for embeddedName, embeddedType := range exactJSONStructFields(embedded) {
					fields[embeddedName] = embeddedType
				}
			}
			continue
		}
		if name == "" {
			name = field.Name
		}
		fields[name] = field.Type
	}
	return fields
}

func dereferenceJSONType(targetType reflect.Type) reflect.Type {
	for targetType != nil && targetType.Kind() == reflect.Pointer {
		targetType = targetType.Elem()
	}
	return targetType
}

// encoding/json replaces unpaired UTF-16 surrogates with U+FFFD. Translation
// must reject that malformed wire input instead of silently changing model
// text, identifiers, tool arguments, or signatures.
func validateJSONUnicodeEscapes(body []byte) error {
	insideString := false
	for index := 0; index < len(body); index++ {
		if body[index] == '"' {
			insideString = !insideString
			continue
		}
		if body[index] != '\\' || !insideString {
			continue
		}
		next, err := advanceJSONUnicodeEscape(body, index)
		if err != nil {
			return err
		}
		index = next
	}
	return nil
}

func advanceJSONUnicodeEscape(body []byte, index int) (int, error) {
	if index+1 >= len(body) {
		return index, nil
	}
	if body[index+1] != 'u' {
		return index + 1, nil
	}
	value, ok := decodeHexQuad(body, index+2)
	if !ok {
		return index, nil
	}
	if value >= 0xdc00 && value <= 0xdfff {
		return index, fmt.Errorf("%w: lone low surrogate", errInvalidJSONUnicode)
	}
	if value < 0xd800 || value > 0xdbff {
		return index + 5, nil
	}
	if !hasLowSurrogateEscape(body, index) {
		return index, fmt.Errorf("%w: high surrogate is not followed by a low surrogate", errInvalidJSONUnicode)
	}
	return index + 11, nil
}

func hasLowSurrogateEscape(body []byte, index int) bool {
	if index+11 >= len(body) || body[index+6] != '\\' || body[index+7] != 'u' {
		return false
	}
	low, validLow := decodeHexQuad(body, index+8)
	return validLow && low >= 0xdc00 && low <= 0xdfff
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
