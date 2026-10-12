package upstream

import (
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

// loadAware prefers the healthy endpoint whose engine reports the fewest
// queued requests, counting the requests this Router has in flight to it so
// picks between two scrapes spread out instead of piling onto one replica.
//
// It trusts telemetry only when it covers the whole host set: if any
// endpoint names no engine, has no fresh observation, or did not report its
// queue depth, the pick falls back to least-request for every endpoint, so a
// replica is never preferred because its load is unknown. An unhealthy
// observation removes its endpoint from the pick but cannot return an
// endpoint that health checks or outlier ejection took out of rotation.
type loadAware struct {
	cluster   string
	telemetry TelemetryReader
	rnd       Rand
	fallback  *leastRequest
}

func (b *loadAware) pick(set *hostSet) *endpoint {
	if len(set.hosts) <= 1 {
		return b.fallback.pick(set)
	}
	ep, outcome := b.pickByTelemetry(set)
	metrics.RecordUpstreamLoadAwarePick(b.cluster, outcome)
	if ep == nil {
		return b.fallback.pick(set)
	}
	return ep
}

func (b *loadAware) pickByTelemetry(set *hostSet) (*endpoint, string) {
	if b.telemetry == nil {
		return nil, loadAwareNoTelemetry
	}
	var best *endpoint
	var bestScore int64
	ties := 0
	for _, host := range set.hosts {
		score, eligible, outcome := b.score(host)
		if outcome != "" {
			return nil, outcome
		}
		if !eligible {
			continue
		}
		switch {
		case best == nil || score < bestScore ||
			(score == bestScore && host.spec.Weight > best.spec.Weight):
			best, bestScore, ties = host, score, 1
		case score == bestScore && host.spec.Weight == best.spec.Weight:
			// Reservoir sampling keeps an equal chance for every tied host.
			ties++
			if b.rnd.Uint64()%uint64(ties) == 0 {
				best = host
			}
		}
	}
	if best == nil {
		return nil, loadAwareAllUnhealthy
	}
	return best, loadAwareTelemetry
}

// score is an endpoint's load. eligible is false for an endpoint its engine
// reports unhealthy; a non-empty outcome means the host set cannot be
// compared and names why.
func (b *loadAware) score(host *endpoint) (score int64, eligible bool, outcome string) {
	id := host.spec.Telemetry
	if id.Engine == "" {
		return 0, false, loadAwareNoEngine
	}
	observation, fresh := b.telemetry.Fresh(id)
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
	return queued + host.active.Load(), true, ""
}
