package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A regression endpoint that was never a classifier answers {"score": x}, not
// a one-element HuggingFace list. Both are read; a bare scalar is not, because
// nothing says which number it is.
func TestDecodeScoreResponse_Shapes(t *testing.T) {
	cases := []struct {
		name   string
		body   string
		want   float64
		wantOK bool
	}{
		{"hf array", `[{"label":"LABEL_0","score":0.42}]`, 0.42, true},
		{"object", `{"score":0.42}`, 0.42, true},
		{"object with label", `{"label":"difficulty","score":7.5}`, 7.5, true},
		{"object, negative units", ` {"score":-0.5}`, -0.5, true},
		// Some runtimes prepend a UTF-8 byte-order mark. Rejecting it would
		// report a wrong response shape and send the operator to rewrite a
		// payload that was already correct.
		{"byte-order mark", "\xef\xbb\xbf{\"score\":0.42}", 0.42, true},
		{"byte-order mark, array", "\xef\xbb\xbf[{\"score\":0.42}]", 0.42, true},
		{"object without score", `{"label":"x"}`, 0, false},
		{"object null score", `{"score":null}`, 0, false},
		{"bare scalar", `0.42`, 0, false},
		{"two entries", `[{"score":0.1},{"score":0.9}]`, 0, false},
		{"empty array", `[]`, 0, false},
		{"empty body", ``, 0, false},
		{"whitespace only", "  \n", 0, false},
	}
	for _, tc := range cases {
		got, err := decodeScoreResponse([]byte(tc.body))
		if tc.wantOK != (err == nil) {
			t.Errorf("%s: err = %v, want ok=%v", tc.name, err, tc.wantOK)
			continue
		}
		if tc.wantOK && got != tc.want {
			t.Errorf("%s: score = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestScoringHTTPBackend_AcceptsTheObjectShape(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"score": 0.73}`))
	}))
	defer server.Close()

	backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-scorer-svc",
	}, time.Second)
	if err != nil {
		t.Fatalf("newScoringHTTPBackend: %v", err)
	}

	score, err := backend.Score(context.Background(), "text")
	if err != nil {
		t.Fatalf("an object-shaped score must be accepted: %v", err)
	}
	if score != 0.73 {
		t.Fatalf("score = %v, want 0.73", score)
	}
}
