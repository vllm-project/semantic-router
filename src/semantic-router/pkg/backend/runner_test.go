package backend

import (
	"context"
	"errors"
	"testing"
	"time"
)

type runnerTestAdapter struct {
	kind    EngineKind
	samples []BackendTelemetry
	err     error
}

func (a runnerTestAdapter) EngineKind() EngineKind {
	return a.kind
}

func (a runnerTestAdapter) Collect(context.Context) ([]BackendTelemetry, error) {
	return a.samples, a.err
}

func TestRunnerCollectOnceWritesTelemetryAndReportsAdapterError(t *testing.T) {
	store := NewStore(time.Second)
	queueDepth := 3
	reported := false
	runner, err := NewRunner(RunnerConfig{
		Store: store,
		Adapters: []TelemetryAdapter{
			runnerTestAdapter{
				kind: EngineKindVLLM,
				samples: []BackendTelemetry{{
					Identity:   BackendIdentity{BackendID: "vllm-a", ModelName: "qwen3"},
					QueueDepth: &queueDepth,
				}},
			},
			runnerTestAdapter{
				kind: EngineKindSGLang,
				err:  errors.New("scrape failed"),
			},
		},
		OnError: func(kind EngineKind, err error) {
			if kind == EngineKindSGLang && err != nil {
				reported = true
			}
		},
	})
	if err != nil {
		t.Fatalf("NewRunner() error = %v", err)
	}

	err = runner.CollectOnce(context.Background())
	if err == nil {
		t.Fatalf("CollectOnce() error = nil, want adapter error")
	}
	if !reported {
		t.Fatalf("expected adapter error callback")
	}
	telemetry, ok := store.Get(BackendIdentity{BackendID: "vllm-a", ModelName: "qwen3"})
	if !ok {
		t.Fatalf("expected telemetry to be written despite another adapter failing")
	}
	if telemetry.QueueDepth == nil || *telemetry.QueueDepth != queueDepth {
		t.Fatalf("QueueDepth = %#v, want %d", telemetry.QueueDepth, queueDepth)
	}
}

func TestNewRunnerRejectsMissingAdapters(t *testing.T) {
	if _, err := NewRunner(RunnerConfig{}); err == nil {
		t.Fatalf("NewRunner() error = nil, want error")
	}
}
