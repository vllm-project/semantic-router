// Package sglang reads SGLang's Prometheus metrics into backend telemetry
// observations.
//
// SGLang's scheduler exports what load-aware balancing reads as gauges:
// sglang:num_queue_reqs is the queue depth, sglang:num_running_reqs the active
// requests, and sglang:token_usage the KV cache usage. Each data-parallel rank
// reports its own series, labelled dp_rank. By default only one scheduler per
// rank reports, but --enable-metrics-for-all-schedulers makes every tensor
// parallel rank report the same values, and each pipeline stage reports too,
// so the adapter keeps one series per dp_rank (the lowest pp_rank, tp_rank and
// moe_ep_rank) before it sums the counts and averages the usage across ranks.
// With priority scheduling the queue gauges also carry a series per priority
// next to the total (priority=""); only the total is read.
//
// The prefix cache hit rate and the latencies are left absent:
// sglang:cache_hit_rate is the rate of the last logged prefill batch, not a
// rate over the scrape window the contract describes.
package sglang

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry/promscrape"
)

// Source names this adapter in observations.
const Source = "sglang/prometheus"

// maxConcurrentScrapes bounds the scrapes one collection round runs at once.
const maxConcurrentScrapes = 16

const defaultRequestTimeout = time.Second

// Register adds the SGLang adapter to a registry.
func Register(r *backendtelemetry.Registry) error {
	return r.Register(backendtelemetry.EngineSGLang, New)
}

// New builds an adapter for the configured SGLang targets.
func New(cfg backendtelemetry.AdapterConfig) (backendtelemetry.Adapter, error) {
	for _, target := range cfg.Targets {
		if target.Identity.Engine != backendtelemetry.EngineSGLang {
			return nil, fmt.Errorf("target %s runs engine %q, not sglang", target.Identity, target.Identity.Engine)
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
	}, nil
}

type adapter struct {
	targets []backendtelemetry.Target
	ttl     time.Duration
	client  *http.Client
	now     func() time.Time
}

func (a *adapter) Engine() backendtelemetry.Engine { return backendtelemetry.EngineSGLang }

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
			select {
			case limit <- struct{}{}:
			case <-ctx.Done():
				results[i].err = ctx.Err()
				return
			}
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
	observation := backendtelemetry.Observation{
		Schema:      backendtelemetry.SchemaVersion,
		Identity:    target.Identity,
		Source:      Source,
		CollectedAt: a.now(),
		TTL:         a.ttl,
	}
	applyGauges(&observation, families)
	return observation, nil
}

func applyGauges(o *backendtelemetry.Observation, families promscrape.Families) {
	if sum, _, ok := perRank(families, "sglang:num_queue_reqs"); ok {
		o.QueueDepth = observedCount(sum)
	}
	if sum, _, ok := perRank(families, "sglang:num_running_reqs"); ok {
		o.ActiveRequests = observedCount(sum)
	}
	// Each data-parallel rank has its own KV pool, so the replica's usage is
	// their mean, as the vLLM adapter does for its engines.
	if sum, ranks, ok := perRank(families, "sglang:token_usage"); ok {
		usage := sum / float64(ranks)
		if usage >= 0 && usage <= 1 {
			o.KVCacheUsage = backendtelemetry.Observed(usage)
		}
	}
}

// rankOrder are the labels that tell the schedulers of one data-parallel rank
// apart; the series with the lowest values is the one SGLang reports by
// default.
var rankOrder = []string{"pp_rank", "tp_rank", "moe_ep_rank"}

// perRank sums one value per data-parallel rank of a gauge family, and also
// returns how many ranks it summed. Series without a dp_rank label form one
// rank. Per-priority series (a non-empty priority label) are skipped.
func perRank(families promscrape.Families, name string) (sum float64, ranks int, ok bool) {
	family, found := families[name]
	if !found {
		return 0, 0, false
	}
	type pick struct {
		order []int64
		value float64
	}
	picks := make(map[string]pick)
	for _, m := range family.GetMetric() {
		value, isGauge := gaugeValue(m)
		if !isGauge || math.IsNaN(value) {
			continue
		}
		labels := labelMap(m)
		if labels["priority"] != "" {
			continue
		}
		order := make([]int64, len(rankOrder))
		for i, label := range rankOrder {
			order[i] = rankNumber(labels[label])
		}
		rank := labels["dp_rank"]
		if current, seen := picks[rank]; !seen || lower(order, current.order) {
			picks[rank] = pick{order: order, value: value}
		}
	}
	for _, p := range picks {
		sum += p.value
	}
	return sum, len(picks), len(picks) > 0
}

func gaugeValue(m *dto.Metric) (float64, bool) {
	switch {
	case m.GetGauge() != nil:
		return m.GetGauge().GetValue(), true
	case m.GetUntyped() != nil:
		return m.GetUntyped().GetValue(), true
	}
	return 0, false
}

func labelMap(m *dto.Metric) map[string]string {
	labels := make(map[string]string, len(m.GetLabel()))
	for _, pair := range m.GetLabel() {
		labels[pair.GetName()] = pair.GetValue()
	}
	return labels
}

// rankNumber reads a rank label; a missing or non-numeric one sorts last.
func rankNumber(value string) int64 {
	n, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return math.MaxInt64
	}
	return n
}

func lower(a, b []int64) bool {
	for i := range a {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// observedCount leaves an invalid gauge absent rather than reporting an idle
// replica.
func observedCount(v float64) backendtelemetry.Count {
	if !(v >= 0 && v < math.MaxInt64) {
		return backendtelemetry.Count{}
	}
	return backendtelemetry.Observed(int64(math.Round(v)))
}
