package classification

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Every token_spans error path may reach operational logs. None of them may
// carry provider text or input text: a provider that echoes the prompt it was
// asked to classify would otherwise put user PII into the log line. These
// cases plant a marker in each place a provider could echo it and assert the
// returned error carries only structural metadata.
func TestTokenSpansErrorsCarryNoProviderOrInputText(t *testing.T) {
	// A marker that stands in for user content a provider might echo back.
	const secret = "MARKER-4711-ssn-987-65-4320" //nolint:gosec // G101: a test marker, not a credential
	input := "call me, my number is " + secret + " thanks"

	cases := []struct {
		name    string
		respond func(inputs string) any
		status  int
		model   string
	}{
		{"error member echoes the prompt", func(inputs string) any {
			return map[string]any{"error": "cannot classify: " + inputs}
		}, 0, ""},
		{"mismatched span text is user content", func(inputs string) any {
			return []map[string]any{{"label": "PERSON", "score": 0.9, "text": secret, "start": 0, "end": 4}}
		}, 0, ""},
		{"conflicting text and word", func(inputs string) any {
			return []map[string]any{{"label": "PERSON", "score": 0.9, "text": secret, "word": secret + "x", "start": 22, "end": 22 + len(secret)}}
		}, 0, ""},
		{"unknown label is provider text", func(inputs string) any {
			return []map[string]any{{"label": secret, "score": 0.9, "text": "call", "start": 0, "end": 4}}
		}, 0, ""},
		{"conflicting label and entity_group", func(inputs string) any {
			return []map[string]any{{"label": "PERSON", "entity_group": secret, "score": 0.9, "text": "call", "start": 0, "end": 4}}
		}, 0, ""},
		{"model identity names a different model", func(inputs string) any {
			return map[string]any{"model": secret, "spans": []any{}}
		}, 0, "expected-model"},
		{"non-2xx body echoes the prompt", func(inputs string) any {
			return "upstream failed on: " + inputs
		}, http.StatusBadGateway, ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg *config.ExternalModelConfig
			if tc.status != 0 {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(tc.status)
					_, _ = w.Write([]byte(tc.respond(input).(string)))
				}))
				t.Cleanup(server.Close)
				_, cfg = newTokenSpansServer(t, func(string) any { return []any{} })
				cfg = tokenSpansConfigFor(cfg, server.URL)
			} else {
				_, cfg = newTokenSpansServer(t, tc.respond)
			}
			if tc.model != "" {
				cfg.ModelName = tc.model
			}
			backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), time.Second)
			if err != nil {
				t.Fatalf("newHTTPTokenClassifierInference: %v", err)
			}
			_, err = backend.ClassifyTokens(input)
			if err == nil {
				t.Fatal("expected an error")
			}
			msg := err.Error()
			for _, leaked := range []string{secret, "my number is", "call me"} {
				if strings.Contains(msg, leaked) {
					t.Fatalf("error echoes %q: %s", leaked, msg)
				}
			}
			if tc.model != "" && !strings.Contains(msg, tc.model) {
				t.Fatalf("configured model name may be reported, got: %s", msg)
			}
		})
	}
}

// tokenSpansConfigFor repoints an external model config at another test server.
func tokenSpansConfigFor(base *config.ExternalModelConfig, serverURL string) *config.ExternalModelConfig {
	u, err := url.Parse(serverURL)
	if err != nil {
		panic(err)
	}
	port, _ := strconv.Atoi(u.Port())
	cfg := *base
	cfg.ModelEndpoint = config.ClassifierVLLMEndpoint{Address: u.Hostname(), Port: port, Protocol: "http"}
	return &cfg
}
