package classification

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// testPIIMapping covers every label the fixtures use. The byte-vs-character
// fixtures deliberately use a BIO-prefixed spelling for one label to prove the
// prefix is stripped on the mapping side as well as the wire side.
func testPIIMapping() *PIIMapping {
	return &PIIMapping{
		LabelToIdx: map[string]int{
			"O": 0, "B-PERSON": 1, "PHONE_NUMBER": 2, "EMAIL_ADDRESS": 3,
			"URL": 4, "ADDRESS": 5, "CREDIT_CARD": 6,
		},
		IdxToLabel: map[string]string{
			"0": "O", "1": "B-PERSON", "2": "PHONE_NUMBER", "3": "EMAIL_ADDRESS",
			"4": "URL", "5": "ADDRESS", "6": "CREDIT_CARD",
		},
	}
}

func TestHTTPTokenClassifierFixtures(t *testing.T) {
	for _, tc := range loadTokenSpansFixtures(t) {
		t.Run(tc.Name, func(t *testing.T) {
			runTokenSpansFixture(t, tc)
		})
	}
}

// runTokenSpansFixture drives one fixture through a real HTTP round trip and
// checks the outcome the fixture declares.
func runTokenSpansFixture(t *testing.T, tc tokenSpansFixtureCase) {
	t.Helper()
	assertFixtureArithmetic(t, tc)

	_, cfg := newTokenSpansServer(t, func(inputs string) any {
		if inputs != tc.Text {
			t.Errorf("provider received %q, want the exact request string", inputs)
		}
		return wireSpans(tc.Spans)
	})
	backend, err := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
	if err != nil {
		t.Fatalf("construct backend: %v", err)
	}
	defer backend.Close()

	entities, err := backend.classifyTokens(context.Background(), tc.Text)
	switch tc.Expect {
	case "accept", "accept_or_truncated_at":
		assertFixtureAccepted(t, tc, entities, err)
	case "reject":
		assertFixtureRejected(t, tc, entities, err)
	default:
		t.Fatalf("unknown expect value %q", tc.Expect)
	}
}

// A provider that cannot read the whole input must say so. The declared
// partial result is returned with ErrTokenSpansTruncated; a span past the
// declared cut is a contract violation, not a partial result.
func TestHTTPTokenClassifierTruncatedAt(t *testing.T) {
	var target tokenSpansFixtureCase
	for _, tc := range loadTokenSpansFixtures(t) {
		if tc.Expect == "accept_or_truncated_at" {
			target = tc
		}
	}
	if target.Name == "" {
		t.Skip("no accept_or_truncated_at fixture")
	}
	cut := target.Spans[0].Start - 1

	t.Run("declared truncation before the entity", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return map[string]any{"spans": []any{}, "truncated_at": cut}
		})
		backend, err := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if err != nil {
			t.Fatal(err)
		}
		entities, err := backend.classifyTokens(context.Background(), target.Text)
		if !errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("want ErrTokenSpansTruncated, got %v", err)
		}
		if len(entities) != 0 {
			t.Fatalf("want no entities, got %d", len(entities))
		}
	})

	t.Run("span after the declared cut is rejected", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return map[string]any{"spans": wireSpans(target.Spans), "truncated_at": cut}
		})
		backend, err := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if err != nil {
			t.Fatal(err)
		}
		_, err = backend.classifyTokens(context.Background(), target.Text)
		if err == nil || errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("want a contract error, got %v", err)
		}
		if !strings.Contains(err.Error(), "truncated_at") {
			t.Fatalf("error should name truncated_at: %v", err)
		}
	})
}

// A provider that answers 200 with something other than a spans array must not
// read as "no PII found" (Xunzhuo's review on #3498).
func TestHTTPTokenClassifierRejectsMalformedEnvelopes(t *testing.T) {
	rejected := []struct {
		name string
		body any
		want string
	}{
		{"empty object", map[string]any{}, "no spans array"},
		{"null body", json.RawMessage("null"), "array or object"},
		{"error member", map[string]any{"error": "model unavailable"}, "reported an error"},
		{"error beside spans", map[string]any{"spans": []any{}, "error": "degraded"}, "reported an error"},
		{"spans null", map[string]any{"spans": nil}, "no spans array"},
		{"spans object", map[string]any{"spans": map[string]any{}}, "must be an array"},
		{"string body", "ok", "array or object"},
	}
	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			entities, err := classifyThrough(t, tc.body, "Call Anna at anna@example.com")
			if err == nil {
				t.Fatalf("want an error, got %d entities", len(entities))
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error should mention %q: %v", tc.want, err)
			}
			if len(entities) != 0 {
				t.Fatalf("no entities may be returned on a rejected response, got %d", len(entities))
			}
		})
	}
}

// An explicit empty list, in either form, is the one legitimate way to say
// "nothing found".
func TestHTTPTokenClassifierAcceptsExplicitEmptyList(t *testing.T) {
	for _, tc := range []struct {
		name string
		body any
	}{
		{"empty envelope list", map[string]any{"spans": []any{}}},
		{"empty bare list", []any{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entities, err := classifyThrough(t, tc.body, "nothing sensitive here")
			if err != nil {
				t.Fatalf("an explicit empty list is a clean result, got %v", err)
			}
			if len(entities) != 0 {
				t.Fatalf("want zero entities, got %d", len(entities))
			}
		})
	}
}

// classifyThrough serves body from a test provider and classifies text against it.
func classifyThrough(t *testing.T, body any, text string) ([]tasks.TokenEntity, error) {
	t.Helper()
	_, cfg := newTokenSpansServer(t, func(string) any { return body })
	backend, err := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
	if err != nil {
		t.Fatal(err)
	}
	return backend.classifyTokens(context.Background(), text)
}

// HuggingFace pipeline spellings are accepted as aliases, and an explicit byte
// pair must agree with the code-point pair.
func TestHTTPTokenClassifierAliasesAndBytePair(t *testing.T) {
	text := "Contact José Alvarez today."
	// "José" here uses a precomposed é (one code point, two bytes).
	start, end := len([]rune("Contact ")), len([]rune("Contact José Alvarez"))
	bStart, bEnd := len("Contact "), len("Contact José Alvarez")

	t.Run("entity_group and word aliases", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{"entity_group": "PERSON", "word": "José Alvarez", "score": 0.9, "start": start, "end": end}}
		})
		backend, _ := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		entities, err := backend.classifyTokens(context.Background(), text)
		if err != nil || len(entities) != 1 {
			t.Fatalf("aliases rejected: %v (%d entities)", err, len(entities))
		}
		if entities[0].Start != bStart || entities[0].End != bEnd {
			t.Fatalf("byte offsets [%d,%d), want [%d,%d)", entities[0].Start, entities[0].End, bStart, bEnd)
		}
	})

	t.Run("agreeing byte pair accepted", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{
				"label": "PERSON", "text": "José Alvarez", "score": 0.9,
				"start": start, "end": end, "byte_start": bStart, "byte_end": bEnd,
			}}
		})
		backend, _ := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if _, err := backend.classifyTokens(context.Background(), text); err != nil {
			t.Fatalf("agreeing byte pair rejected: %v", err)
		}
	})

	t.Run("disagreeing byte pair rejected", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{
				"label": "PERSON", "text": "José Alvarez", "score": 0.9,
				"start": start, "end": end, "byte_start": start, "byte_end": end,
			}}
		})
		backend, _ := newPIIHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if _, err := backend.classifyTokens(context.Background(), text); err == nil {
			t.Fatal("byte pair equal to code-point pair on multi-byte text should be rejected")
		}
	})
}

func TestNewHTTPTokenClassifierInferenceValidation(t *testing.T) {
	good := &config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		ModelName:     "pii-spans",
	}
	if _, err := newPIIHTTPTokenClassifierInference(nil, testPIIMapping(), 0); err == nil {
		t.Error("nil config accepted")
	}
	if _, err := newPIIHTTPTokenClassifierInference(&config.ExternalModelConfig{ModelName: "x"}, testPIIMapping(), 0); err == nil {
		t.Error("missing address accepted")
	}
	if _, err := newPIIHTTPTokenClassifierInference(good, nil, 0); err == nil {
		t.Error("nil mapping accepted")
	}
	if _, err := newPIIHTTPTokenClassifierInference(good, &PIIMapping{}, 0); err == nil {
		t.Error("empty mapping accepted")
	}
	if _, err := newPIIHTTPTokenClassifierInference(good, testPIIMapping(), 0); err != nil {
		t.Errorf("valid config rejected: %v", err)
	}
}
