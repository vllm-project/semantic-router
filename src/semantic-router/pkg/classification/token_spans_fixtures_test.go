package classification

import (
	_ "embed"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
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

// assertFixtureArithmetic checks the fixture's own counts against Go's view of
// the text, so a broken generator cannot pass a broken adapter.
func assertFixtureArithmetic(t *testing.T, tc tokenSpansFixtureCase) {
	t.Helper()
	if got := len([]rune(tc.Text)); got != tc.TextCodePoints {
		t.Fatalf("fixture code points %d, Go counts %d", tc.TextCodePoints, got)
	}
	if got := len(tc.Text); got != tc.TextBytes {
		t.Fatalf("fixture bytes %d, Go counts %d", tc.TextBytes, got)
	}
}

func assertFixtureAccepted(t *testing.T, tc tokenSpansFixtureCase, entities []tasks.TokenEntity, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("expected accept, got error: %v", err)
	}
	if len(entities) != len(tc.Spans) {
		t.Fatalf("got %d entities, want %d", len(entities), len(tc.Spans))
	}
	for i, sp := range tc.Spans {
		assertFixtureSpan(t, tc.Text, i, sp, entities[i])
	}
}

// assertFixtureSpan checks one returned entity against the fixture span: type,
// byte offsets, the byte slice they select, and the score.
func assertFixtureSpan(t *testing.T, text string, i int, sp tokenSpansFixtureSpan, e tasks.TokenEntity) {
	t.Helper()
	if e.EntityType != stripBIOPrefix(sp.Label) {
		t.Errorf("span %d type %q, want %q", i, e.EntityType, sp.Label)
	}
	if e.Start != sp.ByteStart || e.End != sp.ByteEnd {
		t.Errorf("span %d byte offsets [%d,%d), want [%d,%d) (code points [%d,%d))",
			i, e.Start, e.End, sp.ByteStart, sp.ByteEnd, sp.Start, sp.End)
	}
	if text[e.Start:e.End] != sp.Text {
		t.Errorf("span %d byte slice %q, want %q", i, text[e.Start:e.End], sp.Text)
	}
	if e.Confidence != sp.Score {
		t.Errorf("span %d score %v, want %v", i, e.Confidence, sp.Score)
	}
}

func assertFixtureRejected(t *testing.T, tc tokenSpansFixtureCase, entities []tasks.TokenEntity, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected rejection (%s), got %d entities", tc.Note, len(entities))
	}
	if errors.Is(err, ErrTokenSpansTruncated) {
		t.Fatalf("rejection surfaced as truncation: %v", err)
	}
}
