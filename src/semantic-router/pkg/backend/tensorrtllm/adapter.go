package tensorrtllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

const defaultScrapeRequestTimeout = 2 * time.Second

// Adapter collects TensorRT-LLM / Triton Prometheus metrics and normalizes them
// into engine-neutral backend telemetry samples. Triton is one-process-one-
// endpoint, so each configured target maps to exactly one replica.
type Adapter struct {
	targets []backend.AdapterTarget
	client  *http.Client
	ttl     time.Duration
	now     func() time.Time

	// prev caches the last counter snapshot per target key so tier-1 average
	// latency can be derived from counter deltas across scrapes.
	mu   sync.Mutex
	prev map[string]counterSnapshot
}

// NewAdapter creates a TensorRT-LLM telemetry adapter.
func NewAdapter(cfg backend.AdapterConfig) (backend.TelemetryAdapter, error) {
	if len(cfg.Targets) == 0 {
		return nil, fmt.Errorf("tensorrt-llm telemetry adapter requires at least one target")
	}
	for index, target := range cfg.Targets {
		if strings.TrimSpace(target.Identity.BackendID) == "" {
			return nil, fmt.Errorf("tensorrt-llm telemetry target %d requires backend_id", index)
		}
		if strings.TrimSpace(target.Identity.ModelName) == "" {
			return nil, fmt.Errorf("tensorrt-llm telemetry target %d requires model name", index)
		}
		if strings.TrimSpace(target.MetricsEndpoint) == "" {
			return nil, fmt.Errorf("tensorrt-llm telemetry target %d requires metrics endpoint", index)
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
		prev:    map[string]counterSnapshot{},
	}, nil
}

// Register registers the TensorRT-LLM adapter constructor on the package-level
// backend registry.
func Register() error {
	return backend.RegisterAdapter(backend.EngineKindTensorRTLLM, NewAdapter)
}

// EngineKind returns the adapter engine kind.
func (a *Adapter) EngineKind() backend.EngineKind {
	return backend.EngineKindTensorRTLLM
}

// Collect scrapes all configured TensorRT-LLM/Triton targets. Per-target
// failures are returned joined so the runner can log them without discarding
// the successful samples (fail-open).
func (a *Adapter) Collect(ctx context.Context) ([]backend.BackendTelemetry, error) {
	if a == nil {
		return nil, fmt.Errorf("tensorrt-llm telemetry adapter is nil")
	}
	var samples []backend.BackendTelemetry
	var errs []error
	for _, target := range a.targets {
		sample, err := a.collectTarget(ctx, target)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		samples = append(samples, sample)
	}
	return samples, errors.Join(errs...)
}

func (a *Adapter) collectTarget(ctx context.Context, target backend.AdapterTarget) (backend.BackendTelemetry, error) {
	families, err := a.scrape(ctx, target)
	if err != nil {
		return backend.BackendTelemetry{}, err
	}

	key := target.Identity.Normalize().Key()
	a.mu.Lock()
	prev := a.prev[key]
	a.mu.Unlock()

	sample, snapshot, recognized := normalizeTarget(target, families, prev, a.ttl, a.now())
	if !recognized {
		return backend.BackendTelemetry{}, fmt.Errorf("tensorrt-llm telemetry scrape %q had no recognized TensorRT-LLM or Triton metrics", target.MetricsEndpoint)
	}

	a.mu.Lock()
	a.prev[key] = snapshot
	a.mu.Unlock()

	return sample, nil
}

func (a *Adapter) scrape(ctx context.Context, target backend.AdapterTarget) (map[string]*dto.MetricFamily, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.MetricsEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("tensorrt-llm telemetry request %q: %w", target.MetricsEndpoint, err)
	}
	for key, value := range target.Headers {
		req.Header.Set(key, value)
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tensorrt-llm telemetry scrape %q: %w", target.MetricsEndpoint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tensorrt-llm telemetry scrape %q returned HTTP %d", target.MetricsEndpoint, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("tensorrt-llm telemetry read %q: %w", target.MetricsEndpoint, err)
	}

	families, err := parseMetricFamilies(string(body))
	if err != nil {
		return nil, fmt.Errorf("tensorrt-llm telemetry parse %q: %w", target.MetricsEndpoint, err)
	}
	return families, nil
}

func parseMetricFamilies(text string) (map[string]*dto.MetricFamily, error) {
	var parser expfmt.TextParser
	return parser.TextToMetricFamilies(strings.NewReader(text))
}
