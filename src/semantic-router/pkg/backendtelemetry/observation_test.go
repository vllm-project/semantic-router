package backendtelemetry

import (
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

func validObservation() Observation {
	return Observation{
		Schema:      SchemaVersion,
		Identity:    Identity{Model: "qwen3-8b", BackendRef: "gpu-a", Engine: EngineVLLM},
		Source:      "vllm/prometheus",
		CollectedAt: time.Unix(1_000, 0),
	}
}

func TestValueAbsentIsNotZero(t *testing.T) {
	var absent Count
	if v, ok := absent.Get(); ok || v != 0 || absent.Present() {
		t.Fatalf("zero Value = (%d, %v), want absent", v, ok)
	}
	zero := Observed[int64](0)
	if v, ok := zero.Get(); !ok || v != 0 || !zero.Present() {
		t.Fatalf("Observed(0) = (%d, %v), want present zero", v, ok)
	}
}

func TestParseEngine(t *testing.T) {
	for _, engine := range Engines() {
		got, err := ParseEngine(string(engine))
		if err != nil || got != engine {
			t.Fatalf("ParseEngine(%q) = %q, %v", engine, got, err)
		}
	}
	if _, err := ParseEngine("VLLM"); err == nil {
		t.Fatal("ParseEngine accepted a differently cased name")
	}
}

func TestValidateAcceptsMinimalObservation(t *testing.T) {
	if err := validObservation().Validate(); err != nil {
		t.Fatalf("Validate() = %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Observation)
		want   string
	}{
		{"schema", func(o *Observation) { o.Schema = 2 }, "schema 2"},
		{"missing model", func(o *Observation) { o.Identity.Model = "" }, "model is required"},
		{"missing backend ref", func(o *Observation) { o.Identity.BackendRef = "" }, "backend ref is required"},
		{"padded replica", func(o *Observation) { o.Identity.Replica = " r1" }, "surrounding spaces"},
		{"unknown engine", func(o *Observation) { o.Identity.Engine = "triton" }, "unknown engine"},
		{"missing source", func(o *Observation) { o.Source = "" }, "source is required"},
		{"missing collected_at", func(o *Observation) { o.CollectedAt = time.Time{} }, "collected_at"},
		{"negative ttl", func(o *Observation) { o.TTL = -time.Second }, "ttl"},
		{"undefined health", func(o *Observation) { o.Health = 9 }, "health 9"},
		{"negative queue", func(o *Observation) { o.QueueDepth = Observed[int64](-1) }, "queue_depth"},
		{"ratio above one", func(o *Observation) { o.KVCacheUsage = Observed(1.5) }, "kv_cache_usage"},
		{"ratio NaN", func(o *Observation) { o.PrefixCacheHitRate = Observed(math.NaN()) }, "prefix_cache_hit_rate"},
		{"negative latency", func(o *Observation) {
			o.TTFT = Observed(-time.Millisecond)
			o.LatencyWindow = time.Second
		}, "ttft"},
		{"latency without window", func(o *Observation) { o.E2E = Observed(time.Second) }, "latency_window is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o := validObservation()
			tc.mutate(&o)
			err := o.Validate()
			if !errors.Is(err, ErrInvalidObservation) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() = %v, want ErrInvalidObservation containing %q", err, tc.want)
			}
		})
	}
}
