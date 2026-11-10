package upstream

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// healthyPanicPercent is Envoy's healthy panic threshold: below it, a cluster
// balances over all of its endpoints instead of the healthy ones.
const healthyPanicPercent = 50

// endpoint is the runtime state of one backend.
type endpoint struct {
	spec EndpointSpec
	// active counts requests in flight, streamed bodies included.
	active atomic.Int64
	// outlier and health are guarded by the cluster's mutex.
	outlier outlierState
	health  healthState
}

// healthy reports whether neither check holds the endpoint out of rotation.
func (e *endpoint) healthy() bool { return !e.health.failed && !e.outlier.ejected }

// cluster is the runtime of one ClusterSpec. Sets that share an unchanged
// spec share the cluster, so its pool and endpoint state survive a rebuild.
type cluster struct {
	spec     ClusterSpec
	shared   *shared
	pool     *pool
	balancer balancer
	breaker  *breaker
	clock    Clock
	hosts    atomic.Pointer[hostSet]
	refs     atomic.Int32
	// used is set once the cluster serves a request; health checks run on
	// their no-traffic interval until then.
	used atomic.Bool

	mu        sync.Mutex
	endpoints []*endpoint
	outlier   *outlierDetector
	checker   *healthChecker
	closed    bool
}

func newCluster(spec ClusterSpec, sh *shared, rnd Rand, clock Clock) *cluster {
	endpoints := make([]*endpoint, len(spec.Endpoints))
	for i := range spec.Endpoints {
		endpoints[i] = &endpoint{spec: spec.Endpoints[i]}
	}
	connect := spec.Policy.Timeouts.over(builtinPolicy().Timeouts).Connect
	c := &cluster{
		spec:      spec,
		shared:    sh,
		pool:      sh.pools.acquire(poolKeyFor(&spec, connect)),
		balancer:  newBalancer(spec.LBPolicy, rnd),
		breaker:   newBreaker(spec.Name, spec.Breakers),
		clock:     clock,
		endpoints: endpoints,
	}
	if spec.Outlier != nil {
		c.outlier = &outlierDetector{spec: spec.Outlier.withDefaults()}
	}
	if spec.HealthCheck != nil {
		c.checker = newHealthChecker(spec.HealthCheck.withDefaults(), endpoints)
	}
	c.mu.Lock()
	c.refreshHostsLocked()
	c.mu.Unlock()
	sh.retainName(spec.Name)
	return c
}

// start launches the cluster's background checks.
func (c *cluster) start() {
	if c.outlier != nil {
		c.mu.Lock()
		c.outlier.timer = c.clock.AfterFunc(c.outlier.spec.Interval, c.sweep)
		c.mu.Unlock()
	}
	if c.checker != nil {
		c.startHealthChecks()
	}
}

func (c *cluster) pick() *endpoint {
	return c.balancer.pick(c.hosts.Load())
}

// pickAvoiding prefers an endpoint the call has not tried, re-picking up to
// hostSelectionRetries times as Envoy's previous_hosts predicate does, and
// settles for the last pick when every candidate was tried.
func (c *cluster) pickAvoiding(tried []*endpoint) *endpoint {
	ep := c.pick()
	for i := 0; i < hostSelectionRetries && ep != nil && slices.Contains(tried, ep); i++ {
		ep = c.pick()
	}
	return ep
}

// observe reports one attempt's result to outlier detection.
func (c *cluster) observe(ep *endpoint, code int) {
	if c.outlier == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		c.observeLocked(ep, code)
	}
}

// refreshHostsLocked publishes the endpoints the balancer picks from: the
// healthy ones, or all of them when fewer than half are healthy.
func (c *cluster) refreshHostsLocked() {
	healthy := make([]*endpoint, 0, len(c.endpoints))
	for _, ep := range c.endpoints {
		up := ep.healthy()
		if up {
			healthy = append(healthy, ep)
		}
		metrics.SetUpstreamEndpointHealthy(c.spec.Name, ep.spec.Name, up)
	}
	panicking := 100*len(healthy) < healthyPanicPercent*len(c.endpoints)
	metrics.SetUpstreamClusterPanic(c.spec.Name, panicking)
	if panicking {
		healthy = c.endpoints
	}
	if current := c.hosts.Load(); current != nil && sameHosts(current.hosts, healthy) {
		return
	}
	c.hosts.Store(newHostSet(healthy))
}

func sameHosts(a, b []*endpoint) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// warm waits until every endpoint has finished its first health check.
func (c *cluster) warm(ctx context.Context) error {
	if c.checker == nil {
		return nil
	}
	select {
	case <-c.checker.warmed:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *cluster) retain() { c.refs.Add(1) }

// release drops one Set's reference and closes the cluster after the last.
func (c *cluster) release() {
	if c.refs.Add(-1) > 0 {
		return
	}
	c.mu.Lock()
	c.closed = true
	if c.outlier != nil && c.outlier.timer != nil {
		c.outlier.timer.Stop()
	}
	for _, ep := range c.endpoints {
		if ep.health.scheduled != nil {
			ep.health.scheduled.Stop()
		}
	}
	c.mu.Unlock()
	c.shared.pools.release(c.pool)
	c.shared.releaseName(c.spec.Name)
}

// shared is the state a Set hands down to the Sets built from it.
type shared struct {
	pools *poolRegistry
	mu    sync.Mutex
	names map[string]int
}

func newShared(pools *poolRegistry) *shared {
	return &shared{pools: pools, names: map[string]int{}}
}

func (s *shared) retainName(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.names[name]++
}

// releaseName forgets a cluster name once no cluster uses it, and drops its
// metric series with it.
func (s *shared) releaseName(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.names[name]--
	if s.names[name] > 0 {
		return
	}
	delete(s.names, name)
	metrics.DeleteUpstreamCluster(name)
}
