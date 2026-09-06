package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newBenchStubServer returns an httptest server that answers every request with
// a valid non-streaming OpenAI chat completion. It lets the Tier-2 Execute
// benchmarks exercise the real fan-out / collector / synthesis machinery over
// localhost HTTP without any external model, so they measure the looper's
// orchestration overhead (goroutines, JSON, quorum) rather than model quality.
// Numbers include localhost HTTP + JSON cost, so these are throughput baselines,
// not pure-CPU ones (unlike the Tier-1 micro-benchmarks).
func newBenchStubServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-bench",
			"object":  "chat.completion",
			"created": 0,
			"model":   "stub",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": "stub answer"},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
		})
	}))
}

func benchExecuteModelRefs(n int) []config.ModelRef {
	refs := make([]config.ModelRef, n)
	for i := range refs {
		refs[i] = config.ModelRef{Model: fmt.Sprintf("model-%d", i)}
	}
	return refs
}

func benchExecuteOriginal() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("benchmark prompt")},
	}
}

// BenchmarkBase_Execute measures the sequential fan-out over N models via stubbed HTTP.
func BenchmarkBase_Execute(b *testing.B) {
	server := newBenchStubServer()
	defer server.Close()
	looper := NewBaseLooper(&config.LooperConfig{Endpoint: server.URL})

	for _, n := range []int{1, 3, 5} {
		b.Run(fmt.Sprintf("models_%d", n), func(b *testing.B) {
			req := &Request{OriginalRequest: benchExecuteOriginal(), ModelRefs: benchExecuteModelRefs(n), DecisionName: "bench"}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := looper.Execute(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFusion_Execute measures the panel fan-out + judge analysis + final
// synthesis (N+2 model calls) as the panel breadth grows.
func BenchmarkFusion_Execute(b *testing.B) {
	server := newBenchStubServer()
	defer server.Close()
	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})

	for _, n := range []int{2, 3, 5} {
		b.Run(fmt.Sprintf("panel_%d", n), func(b *testing.B) {
			req := &Request{OriginalRequest: benchExecuteOriginal(), ModelRefs: benchExecuteModelRefs(n), DecisionName: "bench"}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := looper.Execute(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkReMoM_Execute measures the multi-round parallel schedule
// (sum(BreadthSchedule)+1 model calls) as rounds/breadth grow.
func BenchmarkReMoM_Execute(b *testing.B) {
	server := newBenchStubServer()
	defer server.Close()
	looper := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL})
	refs := benchExecuteModelRefs(3)

	for _, tc := range []struct {
		label string
		sched []int
	}{
		{"1x4", []int{4}},
		{"2x4", []int{4, 4}},
		{"1x8", []int{8}},
	} {
		b.Run(tc.label, func(b *testing.B) {
			cfg := getDefaultReMoMConfig()
			cfg.BreadthSchedule = tc.sched
			req := &Request{
				OriginalRequest: benchExecuteOriginal(),
				ModelRefs:       refs,
				DecisionName:    "bench",
				Algorithm:       &config.AlgorithmConfig{Type: "remom", ReMoM: cfg},
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := looper.Execute(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFlow_Execute measures a static workflow (one worker step + final
// synthesis) as the worker-step breadth grows. Static mode requires
// MaxParallel >= len(step.Models), so parallelism tracks the worker count.
func BenchmarkFlow_Execute(b *testing.B) {
	server := newBenchStubServer()
	defer server.Close()
	looper := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL})

	for _, n := range []int{1, 2, 3} {
		b.Run(fmt.Sprintf("workers_%d", n), func(b *testing.B) {
			workerModels := make([]string, n)
			for i := range workerModels {
				workerModels[i] = fmt.Sprintf("model-%d", i)
			}
			req := &Request{
				OriginalRequest: benchExecuteOriginal(),
				ModelRefs:       benchExecuteModelRefs(n),
				DecisionName:    "bench",
				Algorithm: &config.AlgorithmConfig{
					Type: "workflows",
					Workflows: &config.WorkflowsAlgorithmConfig{
						Mode:        config.WorkflowModeStatic,
						Roles:       []config.WorkflowRoleConfig{{Name: "worker", Models: workerModels}},
						Final:       config.WorkflowFinalConfig{Model: "model-0"},
						MaxParallel: n,
					},
				},
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := looper.Execute(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
