package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestLLMLabelClassifierRationaleHTTP(t *testing.T) {
	const scoresOnly = `{"scores":{"SAFE":0.25,"RISKY":0.75}}`
	tests := []struct {
		name      string
		flag      string
		response  string
		wantError string
	}{
		{"default requires rationale", "", scoresOnly, "exactly scores and rationale"},
		{"false requires rationale", "false", scoresOnly, "exactly scores and rationale"},
		{"default accepts rationale", "", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":" reason "}`, ""},
		{"false rejects empty rationale", "false", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":" "}`, "empty rationale"},
		{"disabled accepts scores only", "true", scoresOnly, ""},
		{"disabled accepts empty rationale", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":""}`, ""},
		{"disabled accepts rationale", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":"reason"}`, ""},
		{"disabled rejects extra fields", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"extra":true}`, "exactly scores"},
		{"disabled rejects extra field with rationale", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":"","extra":true}`, "exactly scores"},
		{"disabled requires scores", "true", `{"rationale":""}`, "exactly scores"},
		{"disabled rejects missing label", "true", `{"scores":{"SAFE":1}}`, "exactly the declared labels"},
		{"disabled rejects unexpected label", "true", `{"scores":{"SAFE":0.25,"OTHER":0.75}}`, `missing label "RISKY"`},
		{"disabled rejects negative score", "true", `{"scores":{"SAFE":-0.25,"RISKY":1.25}}`, "within [0, 1]"},
		{"disabled rejects incorrect sum", "true", `{"scores":{"SAFE":0.25,"RISKY":0.25}}`, "want approximately 1"},
		{"disabled rejects string score", "true", `{"scores":{"SAFE":"0.25","RISKY":0.75}}`, "invalid JSON"},
		{"disabled rejects null score", "true", `{"scores":{"SAFE":null,"RISKY":1}}`, "must be a number"},
		{"default rejects null score", "", `{"scores":{"SAFE":null,"RISKY":1},"rationale":"reason"}`, "must be a number"},
		{"disabled rejects nonfinite score", "true", `{"scores":{"SAFE":1e999,"RISKY":0}}`, "invalid JSON"},
		{"disabled rejects malformed JSON", "true", `{"scores":`, "invalid JSON"},
		{"disabled rejects rationale type", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":42}`, "invalid JSON"},
		{"disabled rejects null rationale", "true", `{"scores":{"SAFE":0.25,"RISKY":0.75},"rationale":null}`, "rationale must be a string"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requests := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				requests <- body
				if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
					http.Error(w, "unexpected classifier request", http.StatusBadRequest)
					return
				}
				t.Logf("HTTP %s %s -> 200 content=%s", r.Method, r.URL.Path, tt.response)
				w.Header().Set("Content-Type", "application/json")
				content, err := json.Marshal(tt.response)
				if err != nil {
					t.Error(err)
					return
				}
				_, _ = fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%s}}]}`, content)
			}))
			t.Cleanup(server.Close)
			cfg := rationaleHTTPConfig(t, server, tt.flag)
			models, err := newClassifierModelRuntime(cfg, nil)
			if err != nil {
				t.Fatal(err)
			}
			classifier, err := newLLMLabelClassifier(
				models.cfg.ClassifierRules[0], models.cfg.FindExternalModelByName("judge"), models,
			)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { closeLabelClassifiers(map[string]labelClassifier{"risk": classifier}) })
			result, err := classifier.Classify(context.Background(), "Classify this request.")
			t.Logf("disable_rationale=%q scores=%v error=%v", tt.flag, result.Scores, err)
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("error = %v, want %q", err, tt.wantError)
				}
			} else if err != nil || result.Scores["SAFE"] != 0.25 || result.Scores["RISKY"] != 0.75 {
				t.Fatalf("scores = %v, error = %v", result.Scores, err)
			}
			var body []byte
			select {
			case body = <-requests:
			default:
				t.Fatal("classifier did not send an HTTP request")
			}
			t.Logf("request=%s", body)
			var request struct {
				Model          string `json:"model"`
				MaxTokens      int    `json:"max_tokens"`
				ResponseFormat struct {
					Type string `json:"type"`
				} `json:"response_format"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			if request.Model != "test-model" || request.MaxTokens != defaultLLMLabelClassifierMaxTokens || request.ResponseFormat.Type != "json_object" {
				t.Fatalf("unexpected generation options: %s", body)
			}
			if len(request.Messages) != 2 || request.Messages[0].Role != "system" || request.Messages[1].Role != "user" {
				t.Fatalf("unexpected messages: %s", body)
			}
			wantPrompt := "Classify the input.\n\nScore every label from: SAFE, RISKY.\n" +
				`Return only JSON with exactly "scores" and "rationale". ` +
				`"scores" must map every exact label to a number between 0 and 1, ` +
				`and all scores must sum to 1. "rationale" must be a short reason.`
			if tt.flag == "true" {
				wantPrompt = strings.ReplaceAll(wantPrompt, ` and "rationale"`, "")
				wantPrompt = strings.TrimSuffix(wantPrompt, ` "rationale" must be a short reason.`)
			}
			if request.Messages[0].Content != wantPrompt || request.Messages[1].Content != "Classify this request." {
				t.Fatalf("unexpected prompt: %s", body)
			}
		})
	}
}

func rationaleHTTPConfig(t *testing.T, server *httptest.Server, flag string) *config.RouterConfig {
	t.Helper()
	endpoint := endpointForTestServer(t, server)
	setting := ""
	if flag != "" {
		setting = "\n        disable_rationale: " + flag
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.3
global:
  model_catalog:
    external:
      - name: judge
        model_role: classification
        llm_model_name: test-model
        parser_type: json
        llm_endpoint:
          address: %s
          port: %d
          protocol: http
routing:
  signals:
    classifiers:
      - name: risk
        type: llm
        model: judge
        labels: [SAFE, RISKY]
        instructions: Classify the input.%s
`, endpoint.Address, endpoint.Port, setting)))
	if err != nil {
		t.Fatalf("parse canonical classifier configuration: %v", err)
	}
	return cfg
}

func TestLLMLabelClassifierRationaleBindingIsolation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"{\"scores\":{\"SAFE\":0.25,\"RISKY\":0.75}}"}}]}`)
	}))
	t.Cleanup(server.Close)
	pool := binding.NewPool()
	var generations []*Classifier
	for _, flag := range []string{"false", "true", "false"} {
		cfg := rationaleHTTPConfig(t, server, flag)
		other := rationaleHTTPConfig(t, server, "true").ClassifierRules[0]
		other.Name = "other"
		cfg.ClassifierRules = append(cfg.ClassifierRules, other)
		models, err := newClassifierModelRuntime(cfg, native.New(pool))
		if err != nil {
			t.Fatal(err)
		}
		builder := &classifierOptionBuilder{cfg: models.cfg, models: models}
		apply, err := builder.buildGenericClassifiersOption()
		if err != nil {
			t.Fatal(err)
		}
		classifier := &Classifier{}
		apply(classifier)
		generations = append(generations, classifier)
		t.Cleanup(func() { closeLabelClassifiers(classifier.genericClassifiers) })
	}
	for i, generation := range generations {
		for name, classifier := range generation.genericClassifiers {
			result, err := classifier.Classify(context.Background(), "Classify this request.")
			wantRationale := name == "risk" && i != 1
			if wantRationale {
				if err == nil || !strings.Contains(err.Error(), "exactly scores and rationale") {
					t.Fatalf("generation %d, rule %s: error = %v", i, name, err)
				}
			} else if err != nil || result.Scores["RISKY"] != 0.75 {
				t.Fatalf("generation %d, rule %s: scores = %v, error = %v", i, name, result.Scores, err)
			}
		}
	}
}
