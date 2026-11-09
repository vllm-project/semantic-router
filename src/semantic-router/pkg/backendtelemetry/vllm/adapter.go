// Package vllm reads vLLM's Prometheus gauges into backend telemetry
// observations: waiting and running requests and KV cache usage. Optional
// latency and prefix-hit-rate fields stay absent until they have consumers.
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
	}, nil
}

type adapter struct {
	targets []backendtelemetry.Target
	ttl     time.Duration
	client  *http.Client
	now     func() time.Time
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
	at := a.now()
	observation := backendtelemetry.Observation{
		Schema:      backendtelemetry.SchemaVersion,
		Identity:    target.Identity,
		Source:      Source,
		CollectedAt: at,
		TTL:         a.ttl,
	}
	applyGauges(&observation, families)
	return observation, nil
}

func applyGauges(o *backendtelemetry.Observation, families promscrape.Families) {
	if v, _, ok := families.Gauge("vllm:num_requests_waiting"); ok {
		o.QueueDepth = observedCount(v)
	}
	if v, _, ok := families.Gauge("vllm:num_requests_running"); ok {
		o.ActiveRequests = observedCount(v)
	}
	// Each engine of a data-parallel replica reports its own usage, so the
	// replica's usage is their mean.
	if sum, series, ok := families.Gauge("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"); ok {
		usage := sum / float64(series)
		if usage >= 0 && usage <= 1 {
			o.KVCacheUsage = backendtelemetry.Observed(usage)
		}
	}
}

func observedCount(v float64) backendtelemetry.Count {
	if !(v >= 0 && v < math.MaxInt64) {
		return backendtelemetry.Count{}
	}
	return backendtelemetry.Observed(int64(math.Round(v)))
}
