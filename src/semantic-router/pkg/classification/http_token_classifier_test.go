package classification

import (
	_ "embed"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The fixture file is the token_spans.v1 golden set from #2922. Every span
// lists both its code-point offsets (the wire unit) and the byte offsets the
// adapter must produce, so a unit mix-up fails loudly instead of redacting the
// wrong characters.
//
//go:embed testdata/token_spans_v1_fixtures.json
var tokenSpansFixtureJSON []byte

type tokenSpansFixtureSpan struct {
	Label     string  `json:"label"`
	Score     float32 `json:"score"`
	Text      string  `json:"text"`
	Start     int     `json:"start"`
	End       int     `json:"end"`
	ByteStart int     `json:"_byte_start"`
	ByteEnd   int     `json:"_byte_end"`
}

type tokenSpansFixtureCase struct {
	Name           string                  `json:"name"`
	Text           string                  `json:"text"`
	TextCodePoints int                     `json:"text_code_points"`
	TextBytes      int                     `json:"text_bytes"`
	Spans          []tokenSpansFixtureSpan `json:"spans"`
	Expect         string                  `json:"expect"`
	Note           string                  `json:"note"`
}

type tokenSpansFixtureFile struct {
	Contract string                  `json:"contract"`
	Cases    []tokenSpansFixtureCase `json:"cases"`
}

func loadTokenSpansFixtures(t *testing.T) []tokenSpansFixtureCase {
	t.Helper()
	var file tokenSpansFixtureFile
	if err := json.Unmarshal(tokenSpansFixtureJSON, &file); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	if file.Contract != "token_spans.v1" {
		t.Fatalf("fixture contract = %q, want token_spans.v1", file.Contract)
	}
	if len(file.Cases) == 0 {
		t.Fatal("fixture file has no cases")
	}
	return file.Cases
}

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

// wireSpans renders fixture spans in the provider's shape, without the
// underscore-prefixed byte fields the provider does not send.
func wireSpans(spans []tokenSpansFixtureSpan) []map[string]any {
	out := make([]map[string]any, 0, len(spans))
	for _, sp := range spans {
		out = append(out, map[string]any{
			"label": sp.Label, "score": sp.Score, "text": sp.Text, "start": sp.Start, "end": sp.End,
		})
	}
	return out
}

func newTokenSpansServer(t *testing.T, respond func(inputs string) any) (*httptest.Server, *config.ExternalModelConfig) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req httpClassifyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(respond(req.Inputs))
	}))
	t.Cleanup(server.Close)
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server url: %v", err)
	}
	port, _ := strconv.Atoi(u.Port())
	cfg := &config.ExternalModelConfig{
		Name:          "pii-svc",
		ModelRole:     config.ModelRoleClassification,
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: u.Hostname(), Port: port, Protocol: "http"},
		ModelName:     "pii-spans",
	}
	return server, cfg
}

func TestHTTPTokenClassifierFixtures(t *testing.T) {
	cases := loadTokenSpansFixtures(t)
	for _, tc := range cases {
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			// Sanity-check the fixture's own arithmetic against Go's view of
			// the text, so a broken generator cannot pass a broken adapter.
			if got := len([]rune(tc.Text)); got != tc.TextCodePoints {
				t.Fatalf("fixture code points %d, Go counts %d", tc.TextCodePoints, got)
			}
			if got := len(tc.Text); got != tc.TextBytes {
				t.Fatalf("fixture bytes %d, Go counts %d", tc.TextBytes, got)
			}

			_, cfg := newTokenSpansServer(t, func(inputs string) any {
				if inputs != tc.Text {
					t.Errorf("provider received %q, want the exact request string", inputs)
				}
				return wireSpans(tc.Spans)
			})
			backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
			if err != nil {
				t.Fatalf("construct backend: %v", err)
			}
			defer backend.Close()

			entities, err := backend.ClassifyTokens(tc.Text)
			switch tc.Expect {
			case "accept", "accept_or_truncated_at":
				if err != nil {
					t.Fatalf("expected accept, got error: %v", err)
				}
				if len(entities) != len(tc.Spans) {
					t.Fatalf("got %d entities, want %d", len(entities), len(tc.Spans))
				}
				for i, sp := range tc.Spans {
					e := entities[i]
					if e.EntityType != stripBIOPrefix(sp.Label) {
						t.Errorf("span %d type %q, want %q", i, e.EntityType, sp.Label)
					}
					if e.Start != sp.ByteStart || e.End != sp.ByteEnd {
						t.Errorf("span %d byte offsets [%d,%d), want [%d,%d) (code points [%d,%d))",
							i, e.Start, e.End, sp.ByteStart, sp.ByteEnd, sp.Start, sp.End)
					}
					if tc.Text[e.Start:e.End] != sp.Text {
						t.Errorf("span %d byte slice %q, want %q", i, tc.Text[e.Start:e.End], sp.Text)
					}
					if e.Confidence != sp.Score {
						t.Errorf("span %d score %v, want %v", i, e.Confidence, sp.Score)
					}
				}
			case "reject":
				if err == nil {
					t.Fatalf("expected rejection (%s), got %d entities", tc.Note, len(entities))
				}
				if errors.Is(err, ErrTokenSpansTruncated) {
					t.Fatalf("rejection surfaced as truncation: %v", err)
				}
			default:
				t.Fatalf("unknown expect value %q", tc.Expect)
			}
		})
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
		backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if err != nil {
			t.Fatal(err)
		}
		entities, err := backend.ClassifyTokens(target.Text)
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
		backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if err != nil {
			t.Fatal(err)
		}
		_, err = backend.ClassifyTokens(target.Text)
		if err == nil || errors.Is(err, ErrTokenSpansTruncated) {
			t.Fatalf("want a contract error, got %v", err)
		}
		if !strings.Contains(err.Error(), "truncated_at") {
			t.Fatalf("error should name truncated_at: %v", err)
		}
	})
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
		backend, _ := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		entities, err := backend.ClassifyTokens(text)
		if err != nil || len(entities) != 1 {
			t.Fatalf("aliases rejected: %v (%d entities)", err, len(entities))
		}
		if entities[0].Start != bStart || entities[0].End != bEnd {
			t.Fatalf("byte offsets [%d,%d), want [%d,%d)", entities[0].Start, entities[0].End, bStart, bEnd)
		}
	})

	t.Run("agreeing byte pair accepted", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{"label": "PERSON", "text": "José Alvarez", "score": 0.9,
				"start": start, "end": end, "byte_start": bStart, "byte_end": bEnd}}
		})
		backend, _ := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if _, err := backend.ClassifyTokens(text); err != nil {
			t.Fatalf("agreeing byte pair rejected: %v", err)
		}
	})

	t.Run("disagreeing byte pair rejected", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{"label": "PERSON", "text": "José Alvarez", "score": 0.9,
				"start": start, "end": end, "byte_start": start, "byte_end": end}}
		})
		backend, _ := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		if _, err := backend.ClassifyTokens(text); err == nil {
			t.Fatal("byte pair equal to code-point pair on multi-byte text should be rejected")
		}
	})
}

func TestNewHTTPTokenClassifierInferenceValidation(t *testing.T) {
	good := &config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		ModelName:     "pii-spans",
	}
	if _, err := newHTTPTokenClassifierInference(nil, testPIIMapping(), 0); err == nil {
		t.Error("nil config accepted")
	}
	if _, err := newHTTPTokenClassifierInference(&config.ExternalModelConfig{ModelName: "x"}, testPIIMapping(), 0); err == nil {
		t.Error("missing address accepted")
	}
	if _, err := newHTTPTokenClassifierInference(good, nil, 0); err == nil {
		t.Error("nil mapping accepted")
	}
	if _, err := newHTTPTokenClassifierInference(good, &PIIMapping{}, 0); err == nil {
		t.Error("empty mapping accepted")
	}
	if _, err := newHTTPTokenClassifierInference(good, testPIIMapping(), 0); err != nil {
		t.Errorf("valid config rejected: %v", err)
	}
}
