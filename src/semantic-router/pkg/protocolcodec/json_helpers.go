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
	"sync"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var (
	errDuplicateJSONField = errors.New("duplicate JSON field")
	errInvalidJSONUnicode = errors.New("invalid JSON Unicode escape")
	errTrailingJSON       = errors.New("trailing JSON document")
	errNonCanonicalField  = errors.New("non-canonical JSON field")
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
	var exactType reflect.Type
	if rejectUnknownFields(body, policy) {
		exactType = reflect.TypeOf(target)
	}
	if err := validateClientJSONDocument(body, policy, requireObject, exactType); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if exactType != nil {
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

// validateClientJSONDocument also checks member names against exactType when
// it is non-nil; see walkJSONDocument.
func validateClientJSONDocument(body []byte, policy llmprotocol.Policy, requireObject bool, exactType reflect.Type) error {
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
	if err := walkJSONDocument(body, policy.Limits.JSONDepth, exactType); err != nil {
		if errors.Is(err, errDuplicateJSONField) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "duplicate_json_field", "JSON contains a duplicate field", err)
		}
		if errors.Is(err, errTrailingJSON) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "trailing_json", "request body contains trailing JSON", err)
		}
		if errors.Is(err, errNonCanonicalField) {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON contains a non-canonical field", err)
		}
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON is invalid", err)
	}
	return nil
}

func decodeProviderWire(body []byte, target any, policy llmprotocol.Policy) error {
	_, err := decodeProviderJSON(body, target, policy, true)
	return err
}

func decodeProviderJSON(body []byte, target any, policy llmprotocol.Policy, requireObject bool) ([]string, error) {
	_, dropped, err := decodeProviderJSONCanonical(body, target, policy, requireObject)
	return dropped, err
}

// decodeProviderWireVendorAware returns the canonical body alongside any
// extensions removed during decode.
//
// The canonical body matters as much as the field list. Source preservation
// replays preserved bytes verbatim on a same-format encode, so preserving the
// decorated upstream bytes would re-emit decorations the decode just dropped
// and hand a non-canonical body to anything that persists it. Callers that
// build an envelope must build it from these bytes.
func decodeProviderWireVendorAware(body []byte, target any, policy llmprotocol.Policy) ([]byte, []string, error) {
	return decodeProviderJSONCanonical(body, target, policy, true)
}

// decodeProviderValue is the upstream counterpart to decodeWireValue. Provider
// response envelopes remain object-only, while their typed nested arrays and
// scalars use this path without weakening any other JSON validation.
func decodeProviderValue(body []byte, target any, policy llmprotocol.Policy) error {
	_, err := decodeProviderJSON(body, target, policy, false)
	return err
}

func decodeProviderValueVendorAware(body []byte, target any, policy llmprotocol.Policy) ([]string, error) {
	return decodeProviderJSON(body, target, policy, false)
}

func decodeProviderJSONCanonical(body []byte, target any, policy llmprotocol.Policy, requireObject bool) ([]byte, []string, error) {
	// Without vendor extensions the body is decoded as received, so its member
	// names can be checked during the duplicate-key walk.
	var exactType reflect.Type
	if !providerVendorExtensionsAllowed(policy) && rejectUnknownFields(body, policy) {
		exactType = reflect.TypeOf(target)
	}
	if err := validateProviderJSONDocument(body, policy, requireObject, exactType); err != nil {
		return nil, nil, err
	}
	if policy.ResponseVendor == llmprotocol.ResponseVendorOpenAICompatGateway {
		// Anthropic-style usage counters ride inside an otherwise canonical
		// usage object; map them onto the canonical names before the strict
		// decode so the totals check sees a consistent usage object.
		body = translateGatewayUsageCounters(body)
	}
	canonical := body
	decoder := json.NewDecoder(bytes.NewReader(body))
	var dropped []string
	if rejectUnknownFields(body, policy) {
		if exactType == nil {
			canonical, dropped = stripProviderVendorExtensions(body, reflect.TypeOf(target))
			// The stripped body is either the checked original or re-encoded
			// from it, so only its member names can fail here.
			if err := walkJSONDocument(canonical, policy.Limits.JSONDepth, reflect.TypeOf(target)); err != nil {
				return nil, nil, nonCanonicalUpstreamField(err)
			}
		}
		decoder = json.NewDecoder(bytes.NewReader(canonical))
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(target); err != nil {
		return canonical, dropped, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON is invalid", err)
	}
	if err := requireEOF(decoder); err != nil {
		return canonical, dropped, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_trailing_json", "upstream response contains trailing JSON", err)
	}
	return canonical, dropped, nil
}

func nonCanonicalUpstreamField(err error) error {
	// Keep field details in the private cause, not the client message.
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON contains a non-canonical field", err)
}

// validateProviderJSONDocument also checks member names against exactType when
// it is non-nil, like validateClientJSONDocument.
func validateProviderJSONDocument(body []byte, policy llmprotocol.Policy, requireObject bool, exactType reflect.Type) error {
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
	if err := walkJSONDocument(body, policy.Limits.JSONDepth, exactType); err != nil {
		if errors.Is(err, errDuplicateJSONField) {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_duplicate_json_field", "upstream response contains a duplicate field", err)
		}
		if errors.Is(err, errTrailingJSON) {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_trailing_json", "upstream response contains trailing JSON", err)
		}
		if errors.Is(err, errNonCanonicalField) {
			return nonCanonicalUpstreamField(err)
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

// structFieldsByType caches exactJSONStructFields: every request body is
// validated against the same few wire types.
var structFieldsByType sync.Map // reflect.Type -> map[string]reflect.Type

// exactJSONStructFields maps a struct type's JSON member names to their field
// types. The map is shared; callers must not modify it.
func exactJSONStructFields(targetType reflect.Type) map[string]reflect.Type {
	if cached, ok := structFieldsByType.Load(targetType); ok {
		return cached.(map[string]reflect.Type)
	}
	cached, _ := structFieldsByType.LoadOrStore(targetType, buildExactJSONStructFields(targetType))
	return cached.(map[string]reflect.Type)
}

func buildExactJSONStructFields(targetType reflect.Type) map[string]reflect.Type {
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
	return walkJSONDocument(body, maximumDepth, nil)
}

// walkJSONDocument checks duplicate keys, depth and trailing data in one token
// pass. When exactType is set it also checks member names against the wire
// type: encoding/json matches struct fields case-insensitively, so variants
// such as "messAges" would otherwise get past unknown-field rejection. Dynamic
// maps and RawMessage fields stay open and are checked when a codec decodes
// them into a typed union. A name error is only returned once the rest of the
// document passes.
func walkJSONDocument(body []byte, maximumDepth int, exactType reflect.Type) error {
	walker := jsonWalker{decoder: json.NewDecoder(bytes.NewReader(body)), maximumDepth: maximumDepth}
	walker.decoder.UseNumber()
	if err := walker.value(0, exactJSONShape(exactType)); err != nil {
		return err
	}
	if err := requireEOF(walker.decoder); err != nil {
		return fmt.Errorf("%w: %w", errTrailingJSON, err)
	}
	return walker.fieldErr
}

// isJSONObject accepts exactly one top-level JSON object. Streamed tool
// arguments use the same configurable nesting and duplicate-key limits as the
// rest of the wire contract instead of relying on json.Valid's unbounded,
// shape-agnostic check.
func isJSONObject(body []byte, maximumDepth int) bool {
	return llmprotocol.ValidateJSONObject(body, maximumDepth) == nil
}

var jsonRawMessageType = reflect.TypeOf(json.RawMessage{})

// exactJSONShape returns the struct, slice or array type whose member names
// are checked, or nil when any value is accepted.
func exactJSONShape(targetType reflect.Type) reflect.Type {
	targetType = dereferenceJSONType(targetType)
	if targetType == nil || reflect.PointerTo(targetType).Implements(jsonUnmarshalerType) {
		return nil
	}
	switch targetType.Kind() {
	case reflect.Struct:
		return targetType
	case reflect.Slice, reflect.Array:
		if targetType.Elem() == jsonRawMessageType {
			return nil
		}
		return targetType
	}
	return nil
}

type jsonWalker struct {
	decoder      *json.Decoder
	maximumDepth int
	fieldErr     error
}

func (walker *jsonWalker) value(depth int, shape reflect.Type) error {
	if walker.maximumDepth <= 0 || depth > walker.maximumDepth {
		return fmt.Errorf("JSON nesting exceeds the configured limit")
	}
	token, err := walker.decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if shape != nil && token != nil {
		want := json.Delim('[')
		if shape.Kind() == reflect.Struct {
			want = '{'
		}
		if !ok || delimiter != want {
			walker.fieldError(fmt.Errorf("%w: %s value is not %q", errNonCanonicalField, shape, want))
			shape = nil
		}
	}
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		return walker.object(depth, shape)
	case '[':
		return walker.array(depth, shape)
	default:
		return fmt.Errorf("unexpected delimiter %q", delimiter)
	}
}

func (walker *jsonWalker) object(depth int, shape reflect.Type) error {
	var fields map[string]reflect.Type
	if shape != nil {
		fields = exactJSONStructFields(shape)
	}
	seen := make(map[string]struct{})
	for walker.decoder.More() {
		keyToken, err := walker.decoder.Token()
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
		var memberShape reflect.Type
		if fields != nil {
			memberType, known := fields[key]
			if known {
				memberShape = exactJSONShape(memberType)
			} else {
				walker.fieldError(fmt.Errorf("%w: unknown field %q", errNonCanonicalField, key))
			}
		}
		if err := walker.value(depth+1, memberShape); err != nil {
			return err
		}
	}
	closing, err := walker.decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return fmt.Errorf("unterminated object")
	}
	return nil
}

func (walker *jsonWalker) array(depth int, shape reflect.Type) error {
	var elementShape reflect.Type
	if shape != nil {
		elementShape = exactJSONShape(shape.Elem())
	}
	for walker.decoder.More() {
		if err := walker.value(depth+1, elementShape); err != nil {
			return err
		}
	}
	closing, err := walker.decoder.Token()
	if err != nil || closing != json.Delim(']') {
		return fmt.Errorf("unterminated array")
	}
	return nil
}

func (walker *jsonWalker) fieldError(err error) {
	if walker.fieldErr == nil {
		walker.fieldErr = err
	}
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
