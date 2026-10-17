package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// twoPassDecodeWireJSON is decodeWireJSON as it was before the duplicate-key
// walk started checking member names: one walk for duplicates, depth and
// trailing data, then validateExactJSONFieldNames as a separate pass.
func twoPassDecodeWireJSON(body []byte, target any, policy llmprotocol.Policy, requireObject bool) error {
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
	if err := twoPassNoDuplicateKeys(body, policy.Limits.JSONDepth); err != nil {
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

// twoPassNoDuplicateKeys and the consume functions below are the token walk
// as it was before it learned member names, kept here so the reference does
// not share code with the walker under test.
func twoPassNoDuplicateKeys(body []byte, maximumDepth int) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := twoPassConsumeValue(decoder, 0, maximumDepth); err != nil {
		return err
	}
	if err := requireEOF(decoder); err != nil {
		return fmt.Errorf("%w: %w", errTrailingJSON, err)
	}
	return nil
}

func twoPassConsumeValue(decoder *json.Decoder, depth, maximumDepth int) error {
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
		return twoPassConsumeObject(decoder, depth, maximumDepth)
	case '[':
		return twoPassConsumeArray(decoder, depth, maximumDepth)
	default:
		return fmt.Errorf("unexpected delimiter %q", delimiter)
	}
}

func twoPassConsumeObject(decoder *json.Decoder, depth, maximumDepth int) error {
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
		canonicalKey := strings.ToLower(key)
		if _, duplicate := seen[canonicalKey]; duplicate {
			return fmt.Errorf("%w %q", errDuplicateJSONField, key)
		}
		seen[canonicalKey] = struct{}{}
		if err := twoPassConsumeValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return fmt.Errorf("unterminated object")
	}
	return nil
}

func twoPassConsumeArray(decoder *json.Decoder, depth, maximumDepth int) error {
	for decoder.More() {
		if err := twoPassConsumeValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim(']') {
		return fmt.Errorf("unterminated array")
	}
	return nil
}

type wireTarget struct {
	name          string
	newValue      func() any
	requireObject bool
}

var differentialWireTargets = []wireTarget{
	{"chat", func() any { return &chatRequestWire{} }, true},
	{"responses", func() any { return &responsesRequestWire{} }, true},
	{"anthropic", func() any { return &anthropicRequestWire{} }, true},
	{"images", func() any { return &imagesRequestWire{} }, true},
	{"chat messages", func() any { return &[]chatMessageWire{} }, false},
	{"raw values", func() any { return &[]json.RawMessage{} }, false},
}

// outcome reduces a decode result to what callers can observe: the error code
// and message, or the decoded value.
func outcome(value any, err error) string {
	if err != nil {
		var protocolErr *llmprotocol.ProtocolError
		if !errors.As(err, &protocolErr) {
			return "unexpected error type: " + err.Error()
		}
		return "error " + protocolErr.Code + ": " + protocolErr.Message
	}
	encoded, marshalErr := json.Marshal(value)
	if marshalErr != nil {
		return "marshal: " + marshalErr.Error()
	}
	return "ok " + string(encoded)
}

func checkSinglePassMatchesTwoPass(t *testing.T, body []byte) {
	t.Helper()
	for _, policy := range []llmprotocol.Policy{llmprotocol.DefaultPolicy(), shallowPolicy()} {
		for _, target := range differentialWireTargets {
			want := target.newValue()
			wantErr := twoPassDecodeWireJSON(body, want, policy, target.requireObject)
			got := target.newValue()
			gotErr := decodeWireJSON(body, got, policy, target.requireObject)
			if outcome(want, wantErr) != outcome(got, gotErr) {
				t.Fatalf("%s (depth %d) diverged for %q\ntwo-pass:    %s\nsingle-pass: %s",
					target.name, policy.Limits.JSONDepth, body, outcome(want, wantErr), outcome(got, gotErr))
			}
		}
	}
}

func shallowPolicy() llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.JSONDepth = 4
	return policy
}

// differentialSeedBodies returns the golden request bodies, plus the body of
// every rejection fixture: the parsed "body" as written, or the raw
// "body_json" string for inputs that are not valid JSON on their own.
func differentialSeedBodies(t testing.TB) [][]byte {
	t.Helper()
	bodies := [][]byte{agentChatBody(0), agentChatBody(2)}
	bodies = append(bodies, readFixtures(t, "testdata/golden/request/*-in.json")...)
	for _, fixture := range readFixtures(t, "testdata/golden/rejection/*-in.json") {
		var rejection struct {
			Body     json.RawMessage `json:"body"`
			BodyJSON *string         `json:"body_json"`
		}
		if err := json.Unmarshal(fixture, &rejection); err != nil {
			t.Fatal(err)
		}
		switch {
		case rejection.BodyJSON != nil:
			bodies = append(bodies, []byte(*rejection.BodyJSON))
		case len(rejection.Body) > 0:
			bodies = append(bodies, rejection.Body)
		}
	}
	if len(bodies) < 100 {
		t.Fatalf("found only %d seed bodies", len(bodies))
	}
	return bodies
}

func readFixtures(t testing.TB, pattern string) [][]byte {
	t.Helper()
	paths, err := filepath.Glob(pattern)
	if err != nil {
		t.Fatal(err)
	}
	var fixtures [][]byte
	for _, path := range paths {
		fixture, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fixtures = append(fixtures, fixture)
	}
	return fixtures
}

// mutateWireBody derives bodies that hit each rejection path: unknown and
// case-folded members, values of the wrong shape, nulls, duplicate keys,
// trailing data, and combinations where more than one check fails.
func mutateWireBody(body []byte) [][]byte {
	var root any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if decoder.Decode(&root) != nil {
		return nil
	}
	var mutants [][]byte
	var visit func(path []any, node any)
	visit = func(path []any, node any) {
		switch typed := node.(type) {
		case map[string]any:
			mutants = append(mutants, withReplaced(root, path, withMember(typed, "zz_unknown", json.Number("1"))))
			keys := make([]string, 0, len(typed))
			for key := range typed {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				value := typed[key]
				mutants = append(mutants,
					withReplaced(root, path, renamedMember(typed, key, strings.ToUpper(key[:1])+key[1:])),
					withReplaced(root, append(path, key), "text"),
					withReplaced(root, append(path, key), []any{}),
					withReplaced(root, append(path, key), map[string]any{}),
					withReplaced(root, append(path, key), nil),
				)
				visit(append(path, key), value)
			}
		case []any:
			for index, value := range typed {
				visit(append(path, index), value)
			}
		}
	}
	visit(nil, root)
	if bytes.HasPrefix(bytes.TrimSpace(body), []byte("{")) {
		trimmed := bytes.TrimSpace(body)
		mutants = append(mutants,
			append([]byte(`{"zz_unknown":1,`), trimmed[1:]...),
			append([]byte(`{"zz_unknown":1,"model":"a","Model":"b",`), trimmed[1:]...),
			append(append([]byte{}, trimmed...), []byte(` {}`)...),
			append([]byte(`{"zz_unknown":1,`), append(append([]byte{}, trimmed[1:]...), []byte(` []`)...)...),
		)
	}
	return mutants
}

func withMember(object map[string]any, key string, value any) map[string]any {
	copied := make(map[string]any, len(object)+1)
	for existing, existingValue := range object {
		copied[existing] = existingValue
	}
	copied[key] = value
	return copied
}

func renamedMember(object map[string]any, from, to string) map[string]any {
	copied := withMember(object, to, object[from])
	if from != to {
		delete(copied, from)
	}
	return copied
}

// withReplaced marshals root with the node at path replaced by value.
func withReplaced(root any, path []any, value any) []byte {
	var replace func(node any, depth int) any
	replace = func(node any, depth int) any {
		if depth == len(path) {
			return value
		}
		switch typed := node.(type) {
		case map[string]any:
			key := path[depth].(string)
			return withMember(typed, key, replace(typed[key], depth+1))
		case []any:
			index := path[depth].(int)
			copied := append([]any(nil), typed...)
			copied[index] = replace(typed[index], depth+1)
			return copied
		}
		panic(fmt.Sprintf("path %v does not match the document", path))
	}
	encoded, err := json.Marshal(replace(root, 0))
	if err != nil {
		panic(err)
	}
	return encoded
}

func TestSinglePassWireValidationMatchesTwoPass(t *testing.T) {
	checked := 0
	for _, seed := range differentialSeedBodies(t) {
		checkSinglePassMatchesTwoPass(t, seed)
		checked++
		for _, mutant := range mutateWireBody(seed) {
			checkSinglePassMatchesTwoPass(t, mutant)
			checked++
		}
	}
	t.Logf("compared %d bodies", checked)
}

func FuzzSinglePassWireValidationMatchesTwoPass(f *testing.F) {
	for _, seed := range differentialSeedBodies(f) {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		checkSinglePassMatchesTwoPass(t, body)
	})
}
