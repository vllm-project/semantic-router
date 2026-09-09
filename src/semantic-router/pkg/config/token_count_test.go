package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// tokenCountExpectation is what a fixture entry expects from
// TokenCount.Value: exactly one of Value or Error is set.
type tokenCountExpectation struct {
	Value *int64 `json:"value"`
	Error string `json:"error"`
	Note  string `json:"note"`
}

// tokenCountCase is one entry of the fixture's cases: a string handed to
// TokenCount directly.
type tokenCountCase struct {
	Input string `json:"input"`
	tokenCountExpectation
}

// yamlTokenCountCase is one entry of the fixture's yaml_cases: the scalar as
// spelled in a config file, and the text TokenCount receives once the
// Router loader has decoded the file untyped and re-marshalled it.
type yamlTokenCountCase struct {
	Scalar string `json:"scalar"`
	Text   string `json:"text"`
	tokenCountExpectation
}

// tokenCountFixture is testdata/token_count_cases.json, the parsing contract
// shared with the CLI test suite (src/vllm-sr/tests).
type tokenCountFixture struct {
	Cases     []tokenCountCase     `json:"cases"`
	YAMLCases []yamlTokenCountCase `json:"yaml_cases"`
}

func loadTokenCountFixture(t *testing.T) tokenCountFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "token_count_cases.json"))
	if err != nil {
		t.Fatalf("read token count cases: %v", err)
	}
	var fixture tokenCountFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("decode token count cases: %v", err)
	}
	if len(fixture.Cases) == 0 || len(fixture.YAMLCases) == 0 {
		t.Fatal("token count fixture must have cases and yaml_cases")
	}
	return fixture
}

// contextBandYAML renders the document both test suites load for a
// yaml_cases entry: one context rule whose min_tokens is spelled as scalar.
func contextBandYAML(scalar string) string {
	return "routing:\n  signals:\n    context:\n      - name: probe\n        min_tokens: " + scalar + "\n        max_tokens: 100M\n"
}

// decodeTokenCountLikeRouter runs the loader's decode, marshal, decode steps
// (parseYAMLBytesWithOptions) on doc and returns the min_tokens text the
// typed config receives.
func decodeTokenCountLikeRouter(t *testing.T, doc string) TokenCount {
	t.Helper()
	raw, err := parseRawConfigMap([]byte(doc))
	if err != nil {
		t.Fatalf("decode untyped: %v", err)
	}
	remarshalled, err := yaml.Marshal(raw)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	var fragment routingFragmentDocument
	if err := yaml.Unmarshal(remarshalled, &fragment); err != nil {
		t.Fatalf("decode typed: %v", err)
	}
	if len(fragment.Routing.Signals.Context) != 1 {
		t.Fatalf("decoded %d context rules, want 1", len(fragment.Routing.Signals.Context))
	}
	return fragment.Routing.Signals.Context[0].MinTokens
}

// TestTokenCountValueMatchesSharedContract pins TokenCount.Value to the
// fixture the CLI's parse_token_count is also tested against, so the two
// parsers cannot drift apart silently.
func TestTokenCountValueMatchesSharedContract(t *testing.T) {
	for _, tc := range loadTokenCountFixture(t).Cases {
		t.Run(tc.Input, func(t *testing.T) {
			got, err := TokenCount(tc.Input).Value()
			assertTokenCountResult(t, "Value("+tc.Input+")", tc.tokenCountExpectation, got, err)
		})
	}
}

// TestTokenCountYAMLScalarsMatchSharedContract loads each yaml_cases scalar
// the way the Router loads a config file. yaml.v2 types a plain scalar and
// re-emits it before TokenCount sees it, so the fixture's text is what the
// CLI's YAML boundary must reproduce, and ParseYAMLBytes must reach the
// fixture's outcome.
func TestTokenCountYAMLScalarsMatchSharedContract(t *testing.T) {
	for _, tc := range loadTokenCountFixture(t).YAMLCases {
		t.Run(tc.Scalar, func(t *testing.T) {
			doc := contextBandYAML(tc.Scalar)
			text := decodeTokenCountLikeRouter(t, doc)
			if string(text) != tc.Text {
				t.Fatalf("min_tokens: %s decoded as %q, want %q", tc.Scalar, string(text), tc.Text)
			}
			got, err := text.Value()
			assertTokenCountResult(t, "min_tokens: "+tc.Scalar, tc.tokenCountExpectation, got, err)
			assertParsedTokenCount(t, doc, tc)
		})
	}
}

// assertParsedTokenCount checks the end-to-end outcome of ParseYAMLBytes for
// one yaml_cases entry: the parsed band's lower bound, or a load error that
// names min_tokens and the fixture's error.
func assertParsedTokenCount(t *testing.T, doc string, tc yamlTokenCountCase) {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(doc))
	if tc.Error != "" {
		if err == nil || !strings.Contains(err.Error(), "min_tokens: "+tc.Error) {
			t.Fatalf("ParseYAMLBytes(min_tokens: %s) error = %v, want one containing %q", tc.Scalar, err, "min_tokens: "+tc.Error)
		}
		return
	}
	if err != nil {
		t.Fatalf("ParseYAMLBytes(min_tokens: %s): %v", tc.Scalar, err)
	}
	bounds, err := cfg.ContextRules[0].Bounds()
	if err != nil {
		t.Fatalf("Bounds(min_tokens: %s): %v", tc.Scalar, err)
	}
	if int64(bounds.Min) != *tc.Value {
		t.Fatalf("ParseYAMLBytes(min_tokens: %s) min = %d, want %d", tc.Scalar, bounds.Min, *tc.Value)
	}
}

// assertTokenCountResult checks one fixture expectation: an error case must
// fail with the fixture's error prefix, a value case must parse to its value.
func assertTokenCountResult(t *testing.T, label string, want tokenCountExpectation, got int, err error) {
	t.Helper()
	if (want.Value == nil) == (want.Error == "") {
		t.Fatalf("%s: fixture must set exactly one of value or error", label)
	}
	if want.Error != "" {
		assertTokenCountError(t, label, want.Error, got, err)
		return
	}
	if err != nil {
		t.Fatalf("%s returned error %v, want %d", label, err, *want.Value)
	}
	if int64(got) != *want.Value {
		t.Fatalf("%s = %d, want %d", label, got, *want.Value)
	}
}

func assertTokenCountError(t *testing.T, label, wantPrefix string, got int, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s = %d, want error %q", label, got, wantPrefix)
	}
	if !strings.HasPrefix(err.Error(), wantPrefix+":") {
		t.Fatalf("%s error = %q, want prefix %q", label, err.Error(), wantPrefix)
	}
}

// parseContextBand loads doc through ParseYAMLBytes and returns the bounds of
// its only context rule.
func parseContextBand(t *testing.T, doc string) ContextBounds {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if len(cfg.ContextRules) != 1 {
		t.Fatalf("parsed %d context rules, want 1", len(cfg.ContextRules))
	}
	bounds, err := cfg.ContextRules[0].Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	return bounds
}

// TestContextBandTokenCountsExpandEnvironment pins the loader behaviour the
// CLI relies on when it defers a limit that references the environment: the
// reference is expanded before the count is parsed, and an unset variable
// leaves the limit empty.
func TestContextBandTokenCountsExpandEnvironment(t *testing.T) {
	t.Setenv("CTX_PROBE_MIN", "8001")
	if got := parseContextBand(t, contextBandYAML("${CTX_PROBE_MIN}")); got.Min != 8001 {
		t.Fatalf("min_tokens: ${CTX_PROBE_MIN} parsed as %d, want 8001", got.Min)
	}
	doc := "routing:\n  signals:\n    context:\n      - name: probe\n        min_tokens: 8001\n        max_tokens: ${CTX_PROBE_UNSET}\n"
	if got := parseContextBand(t, doc); !got.Unbounded {
		t.Fatalf("max_tokens: ${CTX_PROBE_UNSET} parsed as %+v, want an open-ended band", got)
	}
}

// TestContextBandAnchorIsTypedEverywhere shows that yaml.v2 typing follows
// the scalar, not the field: an anchored 0123 reaches every alias as 83.
func TestContextBandAnchorIsTypedEverywhere(t *testing.T) {
	doc := "routing:\n  signals:\n    context:\n      - name: probe\n        min_tokens: &count 0123\n        max_tokens: 100M\n        description: *count\n"
	cfg, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	rule := cfg.ContextRules[0]
	if string(rule.MinTokens) != "83" || rule.Description != "83" {
		t.Fatalf("min_tokens = %q, description = %q, want both 83", string(rule.MinTokens), rule.Description)
	}
}
