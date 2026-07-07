package vllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

const (
	MetricRequestsWaiting       = "vllm:num_requests_waiting"
	MetricRequestsRunning       = "vllm:num_requests_running"
	MetricKVCacheUsage          = "vllm:kv_cache_usage_perc"
	MetricTimeToFirstToken      = "vllm:time_to_first_token_seconds"           // #nosec G101 -- vLLM Prometheus metric name, not a credential.
	MetricTimePerOutputToken    = "vllm:request_time_per_output_token_seconds" // #nosec G101 -- vLLM Prometheus metric name, not a credential.
	MetricEndToEndLatency       = "vllm:e2e_request_latency_seconds"
	MetricQueueTime             = "vllm:request_queue_time_seconds"
	MetricPrefixCacheQueries    = "vllm:prefix_cache_queries"
	MetricPrefixCacheHits       = "vllm:prefix_cache_hits"
	MetricExternalCacheQueries  = "vllm:external_prefix_cache_queries"
	MetricExternalCacheHits     = "vllm:external_prefix_cache_hits"
	defaultScrapeRequestTimeout = 2 * time.Second
)

// Adapter collects vLLM Prometheus metrics and normalizes them into backend
// telemetry samples.
type Adapter struct {
	targets []backend.AdapterTarget
	client  *http.Client
	ttl     time.Duration
	now     func() time.Time
}

// NewAdapter creates a vLLM telemetry adapter.
func NewAdapter(cfg backend.AdapterConfig) (backend.TelemetryAdapter, error) {
	if len(cfg.Targets) == 0 {
		return nil, fmt.Errorf("vllm telemetry adapter requires at least one target")
	}
	for index, target := range cfg.Targets {
		if strings.TrimSpace(target.Identity.BackendID) == "" {
			return nil, fmt.Errorf("vllm telemetry target %d requires backend_id", index)
		}
		if strings.TrimSpace(target.Identity.ModelName) == "" {
			return nil, fmt.Errorf("vllm telemetry target %d requires model name", index)
		}
		if strings.TrimSpace(target.MetricsEndpoint) == "" {
			return nil, fmt.Errorf("vllm telemetry target %d requires metrics endpoint", index)
		}
	}
	ttl := cfg.TTL
	if ttl <= 0 {
		ttl = backend.DefaultTelemetryTTL
	}
	requestTimeout := cfg.RequestTimeout
	if requestTimeout <= 0 {
		requestTimeout = defaultScrapeRequestTimeout
	}
	return &Adapter{
		targets: append([]backend.AdapterTarget(nil), cfg.Targets...),
		client:  &http.Client{Timeout: requestTimeout},
		ttl:     ttl,
		now:     time.Now,
	}, nil
}

// Register registers the vLLM adapter constructor.
func Register() error {
	return backend.RegisterAdapter(backend.EngineKindVLLM, NewAdapter)
}

// EngineKind returns the adapter engine kind.
func (a *Adapter) EngineKind() backend.EngineKind {
	return backend.EngineKindVLLM
}

// Collect scrapes all configured vLLM targets.
func (a *Adapter) Collect(ctx context.Context) ([]backend.BackendTelemetry, error) {
	if a == nil {
		return nil, fmt.Errorf("vllm telemetry adapter is nil")
	}
	var samples []backend.BackendTelemetry
	var errs []error
	for _, target := range a.targets {
		targetSamples, err := a.collectTarget(ctx, target)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		samples = append(samples, targetSamples...)
	}
	return samples, errors.Join(errs...)
}

func (a *Adapter) collectTarget(ctx context.Context, target backend.AdapterTarget) ([]backend.BackendTelemetry, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.MetricsEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("vllm telemetry request %q: %w", target.MetricsEndpoint, err)
	}
	for key, value := range target.Headers {
		req.Header.Set(key, value)
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vllm telemetry scrape %q: %w", target.MetricsEndpoint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("vllm telemetry scrape %q returned HTTP %d", target.MetricsEndpoint, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("vllm telemetry read %q: %w", target.MetricsEndpoint, err)
	}

	families, err := parseMetricFamilies(string(body))
	if err != nil {
		return nil, fmt.Errorf("vllm telemetry parse %q: %w", target.MetricsEndpoint, err)
	}
	samples := buildTelemetrySamples(target, families, a.ttl, a.now())
	if len(samples) == 0 {
		return nil, fmt.Errorf("vllm telemetry scrape %q had no recognized vLLM metrics", target.MetricsEndpoint)
	}
	return samples, nil
}

func parseMetricFamilies(text string) (map[string]*dto.MetricFamily, error) {
	var parser expfmt.TextParser
	return parser.TextToMetricFamilies(strings.NewReader(text))
}

type sampleBuilder struct {
	identity        backend.BackendIdentity
	telemetry       backend.BackendTelemetry
	recognized      bool
	prefixQueries   float64
	prefixHits      float64
	externalQueries float64
	externalHits    float64
}

type sampleBuilders struct {
	target      backend.AdapterTarget
	ttl         time.Duration
	collectedAt time.Time
	items       map[string]*sampleBuilder
}

func buildTelemetrySamples(target backend.AdapterTarget, families map[string]*dto.MetricFamily, ttl time.Duration, collectedAt time.Time) []backend.BackendTelemetry {
	builders := sampleBuilders{
		target:      target,
		ttl:         ttl,
		collectedAt: collectedAt,
		items:       map[string]*sampleBuilder{},
	}
	builders.applyGauges(families)
	builders.applyHistograms(families)
	builders.applyCounters(families)
	return builders.finalize()
}

func (builders *sampleBuilders) get(metric *dto.Metric) *sampleBuilder {
	replicaID := labelValue(metric, "engine")
	builder, ok := builders.items[replicaID]
	if ok {
		return builder
	}

	identity := builders.target.Identity.Normalize()
	identity.EngineKind = backend.EngineKindVLLM
	identity.ReplicaID = replicaID
	builder = &sampleBuilder{
		identity: identity,
		telemetry: backend.BackendTelemetry{
			Identity:    identity,
			Health:      backend.HealthStateHealthy,
			Confidence:  1,
			CollectedAt: builders.collectedAt,
			TTL:         builders.ttl,
		},
	}
	builders.items[replicaID] = builder
	return builder
}

func (builders *sampleBuilders) applyGauges(families map[string]*dto.MetricFamily) {
	applyGauge(MetricRequestsWaiting, func(t *backend.BackendTelemetry, value float64) {
		queueDepth := int(math.Round(value))
		t.QueueDepth = &queueDepth
	}, families, builders.get)
	applyGauge(MetricRequestsRunning, func(t *backend.BackendTelemetry, value float64) {
		activeRequests := int(math.Round(value))
		t.ActiveRequests = &activeRequests
	}, families, builders.get)
	applyGauge(MetricKVCacheUsage, func(t *backend.BackendTelemetry, value float64) {
		t.KVCachePressure = &value
	}, families, builders.get)
}

func applyGauge(
	name string,
	set func(*backend.BackendTelemetry, float64),
	families map[string]*dto.MetricFamily,
	getBuilder func(*dto.Metric) *sampleBuilder,
) {
	family := families[name]
	if family == nil {
		return
	}
	for _, metric := range family.Metric {
		if metric.GetGauge() == nil {
			continue
		}
		builder := getBuilder(metric)
		set(&builder.telemetry, metric.GetGauge().GetValue())
		builder.recognized = true
	}
}

func (builders *sampleBuilders) applyHistograms(families map[string]*dto.MetricFamily) {
	applyHistogram(families, MetricTimeToFirstToken, builders.get, func(t *backend.BackendTelemetry, snapshot backend.LatencySnapshot) {
		t.Latency.TTFTSeconds = snapshot
	})
	applyHistogram(families, MetricTimePerOutputToken, builders.get, func(t *backend.BackendTelemetry, snapshot backend.LatencySnapshot) {
		t.Latency.TPOTSeconds = snapshot
	})
	applyHistogram(families, MetricEndToEndLatency, builders.get, func(t *backend.BackendTelemetry, snapshot backend.LatencySnapshot) {
		t.Latency.E2ESeconds = snapshot
	})
	applyHistogram(families, MetricQueueTime, builders.get, func(t *backend.BackendTelemetry, snapshot backend.LatencySnapshot) {
		t.Latency.QueueSeconds = snapshot
	})
}

func (builders *sampleBuilders) applyCounters(families map[string]*dto.MetricFamily) {
	applyCounter(families, MetricPrefixCacheQueries, builders.get, func(builder *sampleBuilder, value float64) {
		builder.prefixQueries += value
	})
	applyCounter(families, MetricPrefixCacheHits, builders.get, func(builder *sampleBuilder, value float64) {
		builder.prefixHits += value
	})
	applyCounter(families, MetricExternalCacheQueries, builders.get, func(builder *sampleBuilder, value float64) {
		builder.externalQueries += value
	})
	applyCounter(families, MetricExternalCacheHits, builders.get, func(builder *sampleBuilder, value float64) {
		builder.externalHits += value
	})
}

func (builders *sampleBuilders) finalize() []backend.BackendTelemetry {
	keys := make([]string, 0, len(builders.items))
	for key := range builders.items {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	results := make([]backend.BackendTelemetry, 0, len(keys))
	for _, key := range keys {
		builder := builders.items[key]
		if !builder.recognized {
			continue
		}
		queries := builder.prefixQueries + builder.externalQueries
		if queries > 0 {
			hitRate := (builder.prefixHits + builder.externalHits) / queries
			builder.telemetry.Affinity.PrefixCacheHitRate = &hitRate
		}
		results = append(results, builder.telemetry)
	}
	return results
}

func applyHistogram(
	families map[string]*dto.MetricFamily,
	name string,
	getBuilder func(*dto.Metric) *sampleBuilder,
	set func(*backend.BackendTelemetry, backend.LatencySnapshot),
) {
	family := families[name]
	if family == nil {
		return
	}
	for _, metric := range family.Metric {
		histogram := metric.GetHistogram()
		if histogram == nil {
			continue
		}
		snapshot := histogramSnapshot(histogram)
		builder := getBuilder(metric)
		set(&builder.telemetry, snapshot)
		builder.recognized = true
	}
}

func applyCounter(
	families map[string]*dto.MetricFamily,
	name string,
	getBuilder func(*dto.Metric) *sampleBuilder,
	add func(*sampleBuilder, float64),
) {
	family := families[name]
	if family == nil {
		return
	}
	for _, metric := range family.Metric {
		counter := metric.GetCounter()
		if counter == nil {
			continue
		}
		builder := getBuilder(metric)
		add(builder, counter.GetValue())
		builder.recognized = true
	}
}

func histogramSnapshot(histogram *dto.Histogram) backend.LatencySnapshot {
	return backend.LatencySnapshot{
		P50Seconds: histogramQuantile(0.50, histogram),
		P90Seconds: histogramQuantile(0.90, histogram),
		P95Seconds: histogramQuantile(0.95, histogram),
		P99Seconds: histogramQuantile(0.99, histogram),
	}
}

func histogramQuantile(q float64, histogram *dto.Histogram) *float64 {
	if histogram == nil || histogram.GetSampleCount() == 0 || len(histogram.Bucket) == 0 {
		return nil
	}
	rank := q * float64(histogram.GetSampleCount())
	for _, bucket := range histogram.Bucket {
		if float64(bucket.GetCumulativeCount()) >= rank {
			value := bucket.GetUpperBound()
			return &value
		}
	}
	value := histogram.Bucket[len(histogram.Bucket)-1].GetUpperBound()
	return &value
}

func labelValue(metric *dto.Metric, name string) string {
	for _, pair := range metric.GetLabel() {
		if pair.GetName() == name {
			return pair.GetValue()
		}
	}
	return ""
}
