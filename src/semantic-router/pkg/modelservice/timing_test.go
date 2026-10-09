package modelservice

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestServerTimingReadsTheRuntimesPhasesAndTotal(t *testing.T) {
	var timing exchangeTiming
	timing.read("parse;dur=0.021, tokenize;dur=0.153, queue;dur=0.008, forward;dur=4.871, post;dur=0.034, serialize;dur=0.019, total;dur=5.141")
	want := [otherPhase]time.Duration{21 * time.Microsecond, 153 * time.Microsecond, 8 * time.Microsecond, 4871 * time.Microsecond, 34 * time.Microsecond, 19 * time.Microsecond}
	if !timing.reported || timing.total != 5141*time.Microsecond || timing.phases != want {
		t.Fatalf("timing = %+v", timing)
	}
}

func TestServerTimingSkipsWhatItCannotRead(t *testing.T) {
	header := http.Header{}
	header.Add("Server-Timing", `edge;dur=9, forward;desc="model";dur="2.5", queue, post;dur=NaN, parse;dur=-1`)
	header.Add("Server-Timing", "tokenize;dur=abc, serialize;dur=1e300")
	timing := timedExchange(time.Now(), &http.Response{Header: header})
	if timing.reported || timing.phases != [otherPhase]time.Duration{3: 2500 * time.Microsecond} {
		t.Fatalf("only forward is readable and there is no total: %+v", timing)
	}
	header.Add("Server-Timing", "total;dur=4")
	if timing = timedExchange(time.Now(), &http.Response{Header: header}); !timing.reported || timing.total != 4*time.Millisecond {
		t.Fatalf("a total in a second header value counts: %+v", timing)
	}
}

func BenchmarkServerTimingReadAndRecord(b *testing.B) {
	response := &http.Response{Header: http.Header{"Server-Timing": {"parse;dur=0.021, tokenize;dur=0.153, queue;dur=0.008, forward;dur=4.871, post;dur=0.034, serialize;dur=0.019, total;dur=5.141"}}}
	started := time.Now()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		timedExchange(started, response).record("benchmark", "classify")
	}
}

func TestTransportIsTheExchangeOutsideTheRuntime(t *testing.T) {
	timing := exchangeTiming{elapsed: 10 * time.Millisecond, total: 7 * time.Millisecond, reported: true}
	if timing.transport() != 3*time.Millisecond {
		t.Fatalf("transport = %v", timing.transport())
	}
	timing.total = 12 * time.Millisecond
	if timing.transport() != 0 {
		t.Fatalf("a total above the exchange leaves no transport, not a negative one: %v", timing.transport())
	}
}

// histogram is the sample count and sum of one labelled series of a histogram
// in the default registry (zero when it has none).
func histogram(t *testing.T, name string, labels map[string]string) (uint64, float64) {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			if matches(metric, labels) {
				return metric.GetHistogram().GetSampleCount(), metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return 0, 0
}

func matches(metric *dto.Metric, labels map[string]string) bool {
	found := 0
	for _, pair := range metric.GetLabel() {
		if value, ok := labels[pair.GetName()]; ok {
			if value != pair.GetValue() {
				return false
			}
			found++
		}
	}
	return found == len(labels)
}

const fakeServerTiming = "parse;dur=0.1, tokenize;dur=0.2, queue;dur=0.3, forward;dur=2, post;dur=0.1, serialize;dur=0.1, total;dur=3"

func TestCallsRecordTheirTransportAndTheRuntimesPhases(t *testing.T) {
	timed := runtimetest.New(classifyHead("timed-guard"))
	timed.SetServerTiming(fakeServerTiming)
	timed.SetDelay(5 * time.Millisecond)
	silent := runtimetest.New(classifyHead("silent-guard"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{timed: {"timed-guard"}, silent: {"silent-guard"}})
	call := map[string]string{"deployment": "timed-guard", "surface": "classify"}

	if _, err := lease.Classify(context.Background(), "timed-guard", classifyText("direct")); err != nil {
		t.Fatal(err)
	}
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	Fan(ctx, 2, func(i int) {
		if _, err := lease.Classify(ctx, "timed-guard", classifyText([]string{"bundled one", "bundled two"}[i])); err != nil {
			t.Error(err)
		}
	})
	leave()
	if calls, _ := timed.Bundles(); calls != 1 {
		t.Fatalf("the two bundled calls are one exchange: %d bundles", calls)
	}
	if _, err := lease.Classify(context.Background(), "timed-guard", classifyText("direct")); err != nil {
		t.Fatal(err)
	}

	count, transport := histogram(t, "vsr_model_runtime_transport_seconds", call)
	if count != 3 || transport < 3*0.002 {
		t.Fatalf("three calls reached the runtime, each waiting at least 5 ms for a 3 ms server total: %d calls, %.4f s", count, transport)
	}
	if requests, _ := histogram(t, "vsr_model_runtime_request_duration_seconds", call); requests != 3 {
		t.Fatalf("the cached repeat is neither a runtime call nor timed: %d calls", requests)
	}
	want := map[string]float64{"parse": 0.0001, "tokenize": 0.0002, "queue": 0.0003, "forward": 0.002, "post": 0.0001, "serialize": 0.0001, "other": 0.0002}
	for _, phase := range serverPhases {
		count, sum := histogram(t, "vsr_model_runtime_server_seconds", map[string]string{"deployment": "timed-guard", "surface": "classify", "phase": phase})
		if count != 3 || sum < 3*want[phase]-1e-9 || sum > 3*want[phase]+1e-9 {
			t.Fatalf("phase %s: %d samples summing %.6f s, want 3 of %.4f s", phase, count, sum, want[phase])
		}
	}

	if _, err := lease.Classify(context.Background(), "silent-guard", classifyText("direct")); err != nil {
		t.Fatal(err)
	}
	if count, _ := histogram(t, "vsr_model_runtime_transport_seconds", map[string]string{"deployment": "silent-guard"}); count != 0 {
		t.Fatalf("a runtime that reports no time records no transport: %d", count)
	}
	if requests, _ := histogram(t, "vsr_model_runtime_request_duration_seconds", map[string]string{"deployment": "silent-guard"}); requests != 1 {
		t.Fatalf("its call is still timed: %d", requests)
	}
}

func TestDecisionCallsRecordTheirTransportDirectAndBundled(t *testing.T) {
	decisions := runtimetest.New(runtimetest.Model{ID: "timed-kai"})
	decisions.SetServerTiming(fakeServerTiming)
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{decisions: {"timed-kai"}})
	if _, err := lease.Decide(context.Background(), "timed-kai", sampleRequest("direct")); err != nil {
		t.Fatal(err)
	}
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	if _, err := lease.Decide(ctx, "timed-kai", sampleRequest("bundled")); err != nil {
		t.Fatal(err)
	}
	leave()
	labels := map[string]string{"deployment": "timed-kai", "surface": "decisions", "phase": "forward"}
	if count, sum := histogram(t, "vsr_model_runtime_server_seconds", labels); count != 2 || sum < 0.004-1e-9 || sum > 0.004+1e-9 {
		t.Fatalf("both decision calls carry the runtime's forward: %d samples, %.6f s", count, sum)
	}
	if count, _ := histogram(t, "vsr_model_runtime_transport_seconds", map[string]string{"deployment": "timed-kai", "surface": "decisions"}); count != 2 {
		t.Fatalf("both decision calls record their transport: %d", count)
	}
}
