package upstream

import (
	"math"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Load-aware pick outcomes, the bounded values of the outcome metric label.
const (
	loadAwareTelemetry     = "telemetry"
	loadAwareNoTelemetry   = "fallback_disabled"
	loadAwareNoEngine      = "fallback_no_engine"
	loadAwareStale         = "fallback_stale"
	loadAwareMissingMetric = "fallback_missing_metric"
	loadAwareAllUnhealthy  = "fallback_unhealthy"
)

// loadAware prefers the healthy endpoint with the fewest queued and locally
// active requests. Weight breaks equal-load ties; it is not a traffic quota.
// Telemetry must cover the whole healthy host set, including tried hosts,
// otherwise selection falls back to least-request for the whole pool.
type loadAware struct {
	cluster  string
	rnd      Rand
	fallback *leastRequest
}

func (b *loadAware) pick(set *hostSet, telemetry TelemetryReader) *endpoint {
	return b.pickAvoiding(set, telemetry, nil)
}

func (b *loadAware) pickAvoiding(set *hostSet, telemetry TelemetryReader, tried []*endpoint) *endpoint {
	if len(set.hosts) <= 1 {
		return b.fallback.pick(set, nil)
	}
	ep, outcome := b.pickByTelemetryAvoiding(set, telemetry, tried)
	metrics.RecordUpstreamLoadAwarePick(b.cluster, outcome)
	if ep != nil {
		return ep
	}
	ep = b.fallback.pick(set, nil)
	for i := 0; i < hostSelectionRetries && ep != nil && slices.Contains(tried, ep); i++ {
		ep = b.fallback.pick(set, nil)
	}
	return ep
}

func (b *loadAware) pickByTelemetry(set *hostSet, telemetry TelemetryReader) (*endpoint, string) {
	return b.pickByTelemetryAvoiding(set, telemetry, nil)
}

func (b *loadAware) pickByTelemetryAvoiding(set *hostSet, telemetry TelemetryReader, tried []*endpoint) (*endpoint, string) {
	if telemetry == nil {
		return nil, loadAwareNoTelemetry
	}
	var untried, retried loadChoice
	for _, host := range set.hosts {
		score, eligible, outcome := b.score(host, telemetry)
		if outcome != "" {
			return nil, outcome
		}
		if !eligible {
			continue
		}
		if slices.Contains(tried, host) {
			retried.consider(host, score, b.rnd)
		} else {
			untried.consider(host, score, b.rnd)
		}
	}
	if untried.host != nil {
		return untried.host, loadAwareTelemetry
	}
	if retried.host != nil {
		return retried.host, loadAwareTelemetry
	}
	return nil, loadAwareAllUnhealthy
}

type loadChoice struct {
	host  *endpoint
	score int64
	ties  uint64
}

func (c *loadChoice) consider(host *endpoint, score int64, rnd Rand) {
	switch {
	case c.host == nil || score < c.score || (score == c.score && host.spec.Weight > c.host.spec.Weight):
		c.host, c.score, c.ties = host, score, 1
	case score == c.score && host.spec.Weight == c.host.spec.Weight:
		// Reservoir sampling keeps an equal chance for every tied host.
		c.ties++
		if rnd.Uint64()%c.ties == 0 {
			c.host = host
		}
	}
}

// score excludes an engine-reported unhealthy host. A non-empty outcome means
// the full pool cannot be compared, even if that host was already tried.
func (b *loadAware) score(host *endpoint, telemetry TelemetryReader) (score int64, eligible bool, outcome string) {
	id := host.spec.Telemetry
	if id.Engine == "" {
		return 0, false, loadAwareNoEngine
	}
	observation, fresh := telemetry.Fresh(id)
	if !fresh {
		return 0, false, loadAwareStale
	}
	if observation.Health == backendtelemetry.HealthUnhealthy {
		return 0, false, ""
	}
	queued, ok := observation.QueueDepth.Get()
	if !ok {
		return 0, false, loadAwareMissingMetric
	}
	active := host.active.Load()
	if active > math.MaxInt64-queued {
		return math.MaxInt64, true, ""
	}
	return queued + active, true, ""
}
