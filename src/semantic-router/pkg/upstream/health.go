package upstream

import (
	"context"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// The template's health-check settings and Envoy's defaults for the rest.
const (
	defaultHealthCheckInterval  = 10 * time.Second
	defaultHealthCheckTimeout   = 2 * time.Second
	defaultNoTrafficInterval    = 60 * time.Second
	defaultUnhealthyThreshold   = 3
	defaultHealthyThreshold     = 1
	healthCheckUserAgent        = "vllm-semantic-router/health-check"
	healthCheckBodyDrainLimit   = 64 << 10
	healthCheckOutcomeSuccess   = "success"
	healthCheckOutcomeStatus    = "status"
	healthCheckOutcomeNetwork   = "network"
	healthCheckExpectedResponse = http.StatusOK
)

func (h HealthCheckSpec) withDefaults() HealthCheckSpec {
	h.Interval = pickDuration(h.Interval, defaultHealthCheckInterval)
	h.NoTrafficInterval = pickDuration(h.NoTrafficInterval, defaultNoTrafficInterval)
	h.Timeout = pickDuration(h.Timeout, defaultHealthCheckTimeout)
	h.UnhealthyThreshold = pickInt(h.UnhealthyThreshold, defaultUnhealthyThreshold)
	h.HealthyThreshold = pickInt(h.HealthyThreshold, defaultHealthyThreshold)
	return h
}

// healthState is one endpoint's active-check record. As in Envoy, an endpoint
// of a health-checked cluster starts failed and pending: it serves only
// through the panic threshold until its first check passes. The cluster's
// mutex guards it.
type healthState struct {
	failed    bool
	pending   bool
	passes    int
	failures  int
	scheduled Timer
}

// healthChecker runs a cluster's active checks, one schedule per endpoint.
type healthChecker struct {
	spec HealthCheckSpec
	// warmed closes once every endpoint finished its first check.
	warmed  chan struct{}
	pending int
	once    sync.Once
}

func newHealthChecker(spec HealthCheckSpec, endpoints []*endpoint) *healthChecker {
	h := &healthChecker{spec: spec, warmed: make(chan struct{}), pending: len(endpoints)}
	for _, ep := range endpoints {
		ep.health = healthState{failed: true, pending: true}
	}
	return h
}

// startHealthChecks checks every endpoint at once, then on its interval.
func (c *cluster) startHealthChecks() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, ep := range c.endpoints {
		c.scheduleCheckLocked(ep, 0)
	}
}

func (c *cluster) scheduleCheckLocked(ep *endpoint, after time.Duration) {
	if c.closed {
		return
	}
	ep.health.scheduled = c.clock.AfterFunc(after, func() { c.runCheck(ep) })
}

func (c *cluster) runCheck(ep *endpoint) {
	status, err := c.probe(ep)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.recordCheckLocked(ep, status, err)
	next := c.checker.spec.Interval
	if !c.used.Load() {
		next = c.checker.spec.NoTrafficInterval
	}
	c.scheduleCheckLocked(ep, next)
}

// probe sends one health check through the cluster's own connection pool.
func (c *cluster) probe(ep *endpoint) (int, error) {
	spec := c.checker.spec
	ctx, cancel := context.WithTimeout(context.Background(), spec.Timeout)
	defer cancel()
	u, err := requestURL(ep.spec.Scheme, ep.spec.Address(), spec.Path)
	if err != nil {
		return 0, err
	}
	req := (&http.Request{
		Method: http.MethodGet, URL: u, Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
		Header: http.Header{"User-Agent": {healthCheckUserAgent}}, Host: ep.spec.Authority(),
	}).WithContext(ctx)
	resp, err := c.pool.transport.RoundTrip(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, healthCheckBodyDrainLimit))
	return resp.StatusCode, nil
}

// recordCheckLocked applies one result with Envoy's rules: a pass clears the
// failure count and, once enough passes accrue (one for the first check),
// returns the endpoint to rotation and lifts an outlier ejection; a status
// other than 200 fails it at once; a network failure fails it after the
// unhealthy threshold.
func (c *cluster) recordCheckLocked(ep *endpoint, status int, err error) {
	h := &ep.health
	spec := c.checker.spec
	outcome := healthCheckOutcomeSuccess
	switch {
	case err == nil && status == healthCheckExpectedResponse:
		h.failures = 0
		h.passes++
		if h.failed && (h.pending || h.passes >= spec.HealthyThreshold) {
			h.failed = false
		}
		if ep.outlier.ejected {
			c.unejectLocked(ep)
		}
	case err == nil:
		outcome = healthCheckOutcomeStatus
		h.passes = 0
		h.failed = true
	default:
		outcome = healthCheckOutcomeNetwork
		h.passes = 0
		if !h.failed {
			h.failures++
			h.failed = h.failures >= spec.UnhealthyThreshold
		}
	}
	metrics.RecordUpstreamHealthCheck(c.spec.Name, ep.spec.Name, outcome)
	if h.pending {
		h.pending = false
		c.checker.pending--
		if c.checker.pending == 0 {
			c.checker.once.Do(func() { close(c.checker.warmed) })
		}
	}
	c.refreshHostsLocked()
}
