// Package vllm reads vLLM's Prometheus metrics into backend telemetry
// observations.
//
// Gauges map directly: vllm:num_requests_waiting is the queue depth,
// vllm:num_requests_running the active requests, and vllm:kv_cache_usage_perc
// (vllm:gpu_cache_usage_perc before vLLM 0.10) the KV cache usage. Latencies
// and the prefix cache hit rate are means over the time since the previous
// scrape of the same replica, derived from histogram and counter deltas; a
// replica's first scrape, a counter reset, or a window without requests
// leaves them absent rather than guessed.
package vllm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry/promscrape"
)

// Source names this adapter in observations.
const Source = "vllm/prometheus"

// maxConcurrentScrapes bounds the scrapes one collection round runs at once.
const maxConcurrentScrapes = 16

const defaultRequestTimeout = time.Second

// Register adds the vLLM adapter to a registry.
func Register(r *backendtelemetry.Registry) error {
	return r.Register(backendtelemetry.EngineVLLM, New)
}

// New builds an adapter for the configured vLLM targets.
func New(cfg backendtelemetry.AdapterConfig) (backendtelemetry.Adapter, error) {
	for _, target := range cfg.Targets {
		if target.Identity.Engine != backendtelemetry.EngineVLLM {
			return nil, fmt.Errorf("target %s runs engine %q, not vllm", target.Identity, target.Identity.Engine)
		}
		if target.MetricsURL == "" {
			return nil, fmt.Errorf("target %s has no metrics URL", target.Identity)
		}
	}
	timeout := cfg.RequestTimeout
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	return &adapter{
		targets: append([]backendtelemetry.Target(nil), cfg.Targets...),
		ttl:     cfg.TTL,
		client:  &http.Client{Timeout: timeout},
		now:     time.Now,
		prev:    make(map[backendtelemetry.Identity]counters),
	}, nil
}

type adapter struct {
	targets []backendtelemetry.Target
	ttl     time.Duration
	client  *http.Client
	now     func() time.Time

	mu   sync.Mutex
	prev map[backendtelemetry.Identity]counters
}

func (a *adapter) Engine() backendtelemetry.Engine { return backendtelemetry.EngineVLLM }

// Collect scrapes every target concurrently. It returns the observations of
// the targets it could read and the joined errors of the rest.
func (a *adapter) Collect(ctx context.Context) ([]backendtelemetry.Observation, error) {
	type result struct {
		observation backendtelemetry.Observation
		err         error
	}
	results := make([]result, len(a.targets))
	limit := make(chan struct{}, maxConcurrentScrapes)
	var wg sync.WaitGroup
	for i, target := range a.targets {
		wg.Add(1)
		go func() {
			defer wg.Done()
			limit <- struct{}{}
			defer func() { <-limit }()
			observation, err := a.scrape(ctx, target)
			results[i] = result{observation, err}
		}()
	}
	wg.Wait()
	observations := make([]backendtelemetry.Observation, 0, len(results))
	var errs []error
	for i, r := range results {
		if r.err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", a.targets[i].Identity, r.err))
			continue
		}
		observations = append(observations, r.observation)
	}
	return observations, errors.Join(errs...)
}

func (a *adapter) scrape(ctx context.Context, target backendtelemetry.Target) (backendtelemetry.Observation, error) {
	families, err := promscrape.Fetch(ctx, a.client, target.MetricsURL)
	if err != nil {
		return backendtelemetry.Observation{}, err
	}
	at := a.now()
	observation := backendtelemetry.Observation{
		Schema:      backendtelemetry.SchemaVersion,
		Identity:    target.Identity,
		Source:      Source,
		CollectedAt: at,
		TTL:         a.ttl,
	}
	applyGauges(&observation, families)
	current := readCounters(families, at)
	a.mu.Lock()
	previous, seen := a.prev[target.Identity]
	a.prev[target.Identity] = current
	a.mu.Unlock()
	if seen {
		applyDeltas(&observation, previous, current)
	}
	return observation, nil
}

func applyGauges(o *backendtelemetry.Observation, families promscrape.Families) {
	if v, _, ok := families.Gauge("vllm:num_requests_waiting"); ok {
		o.QueueDepth = backendtelemetry.Observed(roundCount(v))
	}
	if v, _, ok := families.Gauge("vllm:num_requests_running"); ok {
		o.ActiveRequests = backendtelemetry.Observed(roundCount(v))
	}
	// Each engine of a data-parallel replica reports its own usage, so the
	// replica's usage is their mean.
	if sum, series, ok := families.Gauge("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"); ok {
		o.KVCacheUsage = backendtelemetry.Observed(clampRatio(sum / float64(series)))
	}
}

// counter is one cumulative value and whether the scrape reported it.
type counter struct {
	value float64
	ok    bool
}

// histogram is one histogram's cumulative sum and count.
type histogram struct {
	sum, count float64
	ok         bool
}

type counters struct {
	at            time.Time
	ttft, tpot    histogram
	e2e           histogram
	prefixHits    counter
	prefixQueries counter
}

func readCounters(families promscrape.Families, at time.Time) counters {
	read := func(names ...string) histogram {
		sum, count, ok := families.Histogram(names...)
		return histogram{sum: sum, count: count, ok: ok}
	}
	count := func(names ...string) counter {
		value, ok := families.Counter(names...)
		return counter{value: value, ok: ok}
	}
	return counters{
		at:   at,
		ttft: read("vllm:time_to_first_token_seconds"),
		// vLLM renamed the per-output-token latency to inter-token latency.
		tpot:          read("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
		e2e:           read("vllm:e2e_request_latency_seconds"),
		prefixHits:    count("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
		prefixQueries: count("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
	}
}

func applyDeltas(o *backendtelemetry.Observation, previous, current counters) {
	window := current.at.Sub(previous.at)
	if window <= 0 {
		return
	}
	measured := false
	for _, l := range []struct {
		out     *backendtelemetry.Latency
		was, is histogram
	}{
		{&o.TTFT, previous.ttft, current.ttft},
		{&o.TPOT, previous.tpot, current.tpot},
		{&o.E2E, previous.e2e, current.e2e},
	} {
		if mean, ok := meanOver(l.was, l.is); ok {
			*l.out = backendtelemetry.Observed(mean)
			measured = true
		}
	}
	if measured {
		o.LatencyWindow = window
	}
	if rate, ok := ratioOver(previous.prefixHits, current.prefixHits, previous.prefixQueries, current.prefixQueries); ok {
		o.PrefixCacheHitRate = backendtelemetry.Observed(rate)
	}
}

// meanOver is the mean latency of the requests that finished between two
// scrapes. Both scrapes must report the histogram, and its cumulative values
// must not have gone backwards (an engine restart).
func meanOver(was, is histogram) (time.Duration, bool) {
	if !was.ok || !is.ok {
		return 0, false
	}
	dCount, dSum := is.count-was.count, is.sum-was.sum
	if dCount <= 0 || dSum < 0 {
		return 0, false
	}
	return time.Duration(dSum / dCount * float64(time.Second)), true
}

func ratioOver(hitsWas, hitsIs, queriesWas, queriesIs counter) (float64, bool) {
	if !hitsWas.ok || !hitsIs.ok || !queriesWas.ok || !queriesIs.ok {
		return 0, false
	}
	dHits, dQueries := hitsIs.value-hitsWas.value, queriesIs.value-queriesWas.value
	if dQueries <= 0 || dHits < 0 || dHits > dQueries {
		return 0, false
	}
	return dHits / dQueries, true
}

func roundCount(v float64) int64 {
	if v < 0 || math.IsNaN(v) {
		return 0
	}
	return int64(math.Round(v))
}

func clampRatio(v float64) float64 {
	switch {
	case math.IsNaN(v) || v < 0:
		return 0
	case v > 1:
		return 1
	}
	return v
}
