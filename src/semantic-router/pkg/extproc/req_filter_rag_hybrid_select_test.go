package extproc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHybridRetrievalLatencyKeepsWinnerDuration(t *testing.T) {
	if got := hybridRetrievalLatency("parallel", 1.5, 0.2); got != 0.2 {
		t.Fatalf("parallel winner latency = %v", got)
	}
	if got := hybridRetrievalLatency("sequential", 1.5, 0.2); got != 1.5 {
		t.Fatalf("sequential latency = %v, want wrapper duration", got)
	}
	if got := hybridRetrievalLatency("", 1.5, 0.2); got != 1.5 {
		t.Fatalf("default strategy latency = %v", got)
	}
	if got := hybridRetrievalLatency("parallel", 1.5, 0); got != 1.5 {
		t.Fatalf("unset parallel latency = %v", got)
	}
}

func TestMeasureParallelRAGLookupUsesEndToEndDuration(t *testing.T) {
	ctx := &RequestContext{
		RAGSimilarityScore:  0.9,
		RAGRetrievalLatency: 0.001, // A backend's partial HTTP timing.
	}

	result := measureParallelRAGLookup(ctx, func() (string, error) {
		time.Sleep(20 * time.Millisecond)
		return "fallback context", nil
	})

	if result.context != "fallback context" || result.score != 0.9 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.latency < 15*time.Millisecond.Seconds() {
		t.Fatalf("latency = %v, want the complete child lookup duration", result.latency)
	}
}

func TestRetrieveParallelRecordsSelectedChildEndToEndLatency(t *testing.T) {
	const primaryDelay = 100 * time.Millisecond

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		case "/primary":
			time.Sleep(primaryDelay)
			http.Error(w, "primary unavailable", http.StatusServiceUnavailable)
		case "/v1/vector_stores/fallback/search":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"content":"fallback context","score":0.9}]}`))
		default:
			http.NotFound(w, req)
		}
	}))
	defer server.Close()

	ragConfig := &config.RAGPluginConfig{
		Backend: "hybrid",
		BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{
			Primary:  "external_api",
			Fallback: "openai",
			Strategy: "parallel",
			PrimaryConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{
				Endpoint:        server.URL + "/primary",
				RequestFormat:   "custom",
				RequestTemplate: `{"query":"{{.Query}}"}`,
			}),
			FallbackConfig: config.MustStructuredPayload(&config.OpenAIRAGConfig{
				BaseURL:       server.URL,
				APIKey:        "test-key",
				VectorStoreID: "fallback",
			}),
		}),
	}
	ctx := &RequestContext{UserContent: "where is the fallback?"}

	started := time.Now()
	got, err := (&OpenAIRouter{}).retrieveFromHybrid(context.Background(), ctx, ragConfig)
	wrapperElapsed := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	if got != "fallback context" {
		t.Fatalf("context = %q", got)
	}
	if ctx.RAGRetrievalLatency <= 0 {
		t.Fatal("selected child latency was not recorded")
	}
	if ctx.RAGRetrievalLatency >= (wrapperElapsed / 2).Seconds() {
		t.Fatalf("selected child latency = %v, wrapper latency = %v", ctx.RAGRetrievalLatency, wrapperElapsed)
	}
}

func TestSequentialHybridKeepsWrapperLatency(t *testing.T) {
	sequential := &config.RAGPluginConfig{
		Backend: "hybrid",
		BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{
			Primary:  "milvus",
			Fallback: "qdrant",
		}),
	}
	if got := hybridRetrievalLatency(hybridStrategy(sequential), 1.5, 0.2); got != 1.5 {
		t.Fatalf("sequential config latency = %v", got)
	}

	parallel := &config.RAGPluginConfig{
		Backend: "hybrid",
		BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{
			Primary:  "milvus",
			Fallback: "qdrant",
			Strategy: "parallel",
		}),
	}
	if got := hybridRetrievalLatency(hybridStrategy(parallel), 1.5, 0.2); got != 0.2 {
		t.Fatalf("parallel config latency = %v", got)
	}
}

func TestSelectParallelRAG_PrimaryDoesNotWaitForFallback(t *testing.T) {
	primary := make(chan parallelRAGResult, 1)
	fallback := make(chan parallelRAGResult, 1)
	primary <- parallelRAGResult{context: "primary", score: 0.9}
	started := time.Now()
	lateSend := make(chan struct{})
	go func() {
		time.Sleep(200 * time.Millisecond)
		fallback <- parallelRAGResult{context: "fallback"}
		close(lateSend)
	}()

	got, err := selectParallelRAG(primary, true, fallback)
	if err != nil {
		t.Fatal(err)
	}
	if got.context != "primary" || got.score != 0.9 {
		t.Fatalf("got %#v", got)
	}
	if time.Since(started) > 100*time.Millisecond {
		t.Fatal("waited for the slow fallback")
	}
	select {
	case <-lateSend:
	case <-time.After(time.Second):
		t.Fatal("late fallback send did not finish")
	}
}

func TestSelectParallelRAG_FallbackUsedWhenPrimaryFails(t *testing.T) {
	primary := make(chan parallelRAGResult, 1)
	fallback := make(chan parallelRAGResult, 1)
	primary <- parallelRAGResult{err: errors.New("primary down")}
	fallback <- parallelRAGResult{context: "fallback", score: 0.4}

	got, err := selectParallelRAG(primary, true, fallback)
	if err != nil {
		t.Fatal(err)
	}
	if got.context != "fallback" || got.score != 0.4 {
		t.Fatalf("got %#v", got)
	}
}

func TestSelectParallelRAG_EmptyPrimaryUsesFallback(t *testing.T) {
	primary := make(chan parallelRAGResult, 1)
	fallback := make(chan parallelRAGResult, 1)
	primary <- parallelRAGResult{}
	fallback <- parallelRAGResult{context: "fallback"}

	got, err := selectParallelRAG(primary, true, fallback)
	if err != nil {
		t.Fatal(err)
	}
	if got.context != "fallback" {
		t.Fatalf("got %#v", got)
	}
}

func TestSelectParallelRAG_PrimaryWinsWhenFallbackFinishesFirst(t *testing.T) {
	primary := make(chan parallelRAGResult, 1)
	fallback := make(chan parallelRAGResult, 1)
	fallback <- parallelRAGResult{context: "fallback"}
	go func() {
		time.Sleep(40 * time.Millisecond)
		primary <- parallelRAGResult{context: "primary"}
	}()

	got, err := selectParallelRAG(primary, true, fallback)
	if err != nil {
		t.Fatal(err)
	}
	if got.context != "primary" {
		t.Fatalf("got %#v", got)
	}
}

func TestSelectParallelRAG_BothFailuresKeepWrappedCauses(t *testing.T) {
	primary := make(chan parallelRAGResult, 1)
	fallback := make(chan parallelRAGResult, 1)
	primaryCause := errors.New("primary down")
	fallbackCause := errors.New("fallback down")
	primary <- parallelRAGResult{err: primaryCause}
	fallback <- parallelRAGResult{err: fallbackCause}

	_, err := selectParallelRAG(primary, true, fallback)
	if !errors.Is(err, primaryCause) || !errors.Is(err, fallbackCause) {
		t.Fatalf("causes were not wrapped: %v", err)
	}
}
