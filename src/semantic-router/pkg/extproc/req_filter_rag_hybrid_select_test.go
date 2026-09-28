package extproc

import (
	"errors"
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
