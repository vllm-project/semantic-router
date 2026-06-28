package selection

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLLMRouterPromptRenderer_RendersInlineTemplate(t *testing.T) {
	renderer, err := NewLLMRouterPromptRenderer(&RLDrivenConfig{
		LLMRouterQueryTemplate: `Decision={{.decision_name}}|Domains={{join .matched_domains ","}}|Models={{range .available_models}}{{.name}} {{end}}`,
	})
	if err != nil {
		t.Fatalf("NewLLMRouterPromptRenderer() error = %v", err)
	}

	selCtx := &SelectionContext{
		DecisionName:    "route-a",
		MatchedDomains:  []string{"math", "coding"},
		CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
	}

	got, err := renderer.Render(selCtx)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}

	if !strings.Contains(got, "Decision=route-a") {
		t.Fatalf("rendered prompt missing decision name: %q", got)
	}
	if !strings.Contains(got, "Domains=math,coding") {
		t.Fatalf("rendered prompt missing matched domains: %q", got)
	}
	if !strings.Contains(got, "Models=model-a model-b") {
		t.Fatalf("rendered prompt missing candidate models: %q", got)
	}
}

func TestLLMRouterPromptRenderer_LoadsFromFile(t *testing.T) {
	tmpFile, err := os.CreateTemp(t.TempDir(), "llm-router-query-*.tmpl")
	if err != nil {
		t.Fatalf("CreateTemp() error = %v", err)
	}
	if _, writeErr := tmpFile.WriteString(`Decision={{.decision_name}}`); writeErr != nil {
		t.Fatalf("WriteString() error = %v", writeErr)
	}
	if closeErr := tmpFile.Close(); closeErr != nil {
		t.Fatalf("Close() error = %v", closeErr)
	}

	renderer, err := NewLLMRouterPromptRenderer(&RLDrivenConfig{
		LLMRouterQueryTemplateFile: tmpFile.Name(),
	})
	if err != nil {
		t.Fatalf("NewLLMRouterPromptRenderer() error = %v", err)
	}

	got, err := renderer.Render(&SelectionContext{DecisionName: "route-b"})
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	if got != "Decision=route-b" {
		t.Fatalf("Render() = %q, want %q", got, "Decision=route-b")
	}
}

func TestRLDrivenSelectorSelectWithLLMRouter_UsesRenderedTemplate(t *testing.T) {
	var receivedQuery string
	var handlerErr error
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(done)
		defer r.Body.Close()

		body, err := io.ReadAll(r.Body)
		if err != nil {
			handlerErr = err
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		var payload map[string]string
		if err := json.Unmarshal(body, &payload); err != nil {
			handlerErr = err
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		receivedQuery = payload["query"]
		_ = json.NewEncoder(w).Encode(map[string]string{
			"selected_model": "model-a",
			"thinking":       "picked model-a",
			"full_response":  "route decision",
		})
	}))
	defer server.Close()

	renderer, err := NewLLMRouterPromptRenderer(&RLDrivenConfig{
		LLMRouterQueryTemplate: `Decision={{.decision_name}}|Description={{.decision_description}}|Domains={{join .matched_domains ","}}|Keywords={{join .matched_keywords ","}}|Labels={{join .labels ","}}|Models={{range .available_models}}{{.name}} {{end}}`,
	})
	if err != nil {
		t.Fatalf("NewLLMRouterPromptRenderer() error = %v", err)
	}

	selector := &RLDrivenSelector{
		llmRouterClient:         NewLLMRouterClient(server.URL),
		llmRouterPromptRenderer: renderer,
	}

	selCtx := &SelectionContext{
		Query:               "how do I route this?",
		DecisionName:        "route-a",
		DecisionDescription: "A test decision",
		MatchedDomains:      []string{"math"},
		MatchedKeywords:     []string{"vector"},
		Labels:              []string{"domain:math", "keyword:vector"},
		CandidateModels:     []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		UserID:              "user-123",
		SessionID:           "session-456",
		RequestID:           "req-789",
		ConversationHistory: []string{"previous turn"},
		AgenticSession:      &AgenticSessionContext{PreviousModel: "model-z", PreviousResponseID: "resp-1", TurnIndex: 2, HistoryTokens: 128, CacheWarmth: 0.42, CacheWarmthOK: true, IdleFor: 5 * time.Second, IdleKnown: true},
	}

	result, err := selector.selectWithLLMRouter(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("selectWithLLMRouter() error = %v", err)
	}
	<-done
	if handlerErr != nil {
		t.Fatalf("handler error = %v", handlerErr)
	}
	if result.SelectedModel != "model-a" {
		t.Fatalf("SelectedModel = %q, want model-a", result.SelectedModel)
	}

	assertContainsAll(t, receivedQuery, map[string]string{
		"decision name":        "Decision=route-a",
		"decision description": "Description=A test decision",
		"matched domains":      "Domains=math",
		"matched keywords":     "Keywords=vector",
		"labels":               "Labels=domain:math,keyword:vector",
		"candidate models":     "Models=model-a model-b",
	})
}

func assertContainsAll(t *testing.T, got string, expectations map[string]string) {
	t.Helper()

	for label, want := range expectations {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered query missing %s: %q", label, got)
		}
	}
}
