package extproc

import (
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func newObservedEventLogger(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.InfoLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(restore)
	return logs
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
