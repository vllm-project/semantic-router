package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

func newObservedEventLogger(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.InfoLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(restore)
	return logs
}

func TestLogSignalPhaseTimingIncludesOnlyEvaluatedSignals(t *testing.T) {
	logs := newObservedEventLogger(t)

	signals := &classification.SignalResults{Metrics: &classification.SignalMetricsCollection{}}
	signals.Metrics.Domain.ExecutionTimeMs = 12.5
	signals.Metrics.Complexity.ExecutionTimeMs = 3
	signals.Metrics.Embedding.ExecutionTimeMs = 7.25

	logSignalPhaseTiming(&RequestContext{RequestID: "req-timing-1"}, 40, signals)

	entries := logs.FilterMessage("signal_phase_timing").All()
	if len(entries) != 1 {
		t.Fatalf("expected one signal_phase_timing entry, got %d", len(entries))
	}
	fields := entries[0].ContextMap()
	if got := fields["request_id"]; got != "req-timing-1" {
		t.Errorf("request_id = %v, want req-timing-1", got)
	}
	if got, ok := fields["signal_phase_ms"].(int64); !ok || got != 40 {
		t.Errorf("signal_phase_ms = %v, want 40", fields["signal_phase_ms"])
	}
	for _, want := range []struct {
		key string
		val float64
	}{
		{"domain_signal_ms", 12.5},
		{"complexity_signal_ms", 3},
		{"embedding_signal_ms", 7.25},
	} {
		got, ok := fields[want.key].(float64)
		if !ok {
			t.Errorf("%s missing or not a float64: %v", want.key, fields[want.key])
			continue
		}
		if got != want.val {
			t.Errorf("%s = %v, want %v", want.key, got, want.val)
		}
	}
	// Unevaluated signals (0 ms) must be omitted to keep the line compact.
	for _, unevaluated := range []string{"keyword_signal_ms", "jailbreak_signal_ms", "pii_signal_ms"} {
		if _, ok := fields[unevaluated]; ok {
			t.Errorf("unevaluated signal %s should be omitted, got %v", unevaluated, fields[unevaluated])
		}
	}
}

func TestLogSignalPhaseTimingNilMetricsAndNilSignals(t *testing.T) {
	logs := newObservedEventLogger(t)

	// Must not panic and must still emit the phase wall without per-signal fields.
	logSignalPhaseTiming(&RequestContext{RequestID: "req-nil-metrics"}, 33, &classification.SignalResults{})
	logSignalPhaseTiming(&RequestContext{RequestID: "req-nil-signals"}, 44, nil)

	entries := logs.FilterMessage("signal_phase_timing").All()
	if len(entries) != 2 {
		t.Fatalf("expected 2 signal_phase_timing entries, got %d", len(entries))
	}
	for i, entry := range entries {
		fields := entry.ContextMap()
		if _, ok := fields["domain_signal_ms"]; ok {
			t.Errorf("entry %d: domain_signal_ms must be absent without metrics, got %v", i, fields["domain_signal_ms"])
		}
		if _, ok := fields["signal_phase_ms"]; !ok {
			t.Errorf("entry %d: signal_phase_ms is required even without metrics", i)
		}
	}
}
