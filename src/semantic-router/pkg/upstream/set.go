package upstream

import (
	"context"
	"crypto/x509"
	"fmt"
	"sync"
	"time"

	"golang.org/x/net/http/httpguts"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Options configure how a Set is built. The zero value is the production
// configuration.
type Options struct {
	// Previous is the Set this one replaces. Clusters whose spec did not
	// change carry over with their connection pools and endpoint state, and
	// so do the pools of unchanged security domains; RootCAs and Dial are then
	// inherited from Previous. The caller still closes Previous.
	Previous *Set
	// RootCAs replaces the system certificate pool for upstream TLS.
	RootCAs *x509.CertPool
	// Dial opens connections; nil uses a net.Dialer.
	Dial DialFunc
	// Rand drives load-balancer randomness; nil uses the global source.
	Rand Rand
	// Clock drives health checks and outlier sweeps; nil uses real time.
	Clock Clock
}

// Set serves every cluster of one topology. It is safe for concurrent use.
type Set struct {
	topology  Topology
	clusters  map[string]*cluster
	fallback  *cluster
	listeners map[string]Policy
	first     Policy
	shared    *shared
	rnd       Rand
	clock     Clock

	abort  context.Context
	cancel context.CancelFunc

	mu       sync.Mutex
	closed   bool
	released bool
	inflight int
	drained  chan struct{}
}

// New builds a Set that serves topology.
func New(topology Topology, opts Options) (*Set, error) {
	if err := topology.validate(); err != nil {
		return nil, err
	}
	s := &Set{
		topology:  topology,
		clusters:  make(map[string]*cluster, len(topology.Clusters)),
		listeners: make(map[string]Policy, len(topology.Listeners)),
		drained:   make(chan struct{}),
	}
	s.abort, s.cancel = context.WithCancel(context.Background())
	for i, listener := range topology.Listeners {
		policy := Policy{Timeouts: listener.Timeouts}
		s.listeners[listener.Name] = policy
		if i == 0 {
			s.first = policy
		}
	}
	rnd := opts.Rand
	if rnd == nil {
		rnd = globalRand{}
	} else {
		rnd = &lockedRand{src: rnd}
	}
	clock := opts.Clock
	if clock == nil {
		clock = realClock{}
	}
	s.rnd, s.clock = rnd, clock
	previous := opts.Previous.retainClusters()
	if opts.Previous != nil {
		s.shared = opts.Previous.shared
	} else {
		s.shared = newShared(newPoolRegistry(opts.RootCAs, opts.Dial))
	}
	var fresh []*cluster
	for _, spec := range topology.Clusters {
		c := previous[spec.Name]
		if c == nil || !c.spec.equal(&spec) {
			c = newCluster(spec, s.shared, rnd, clock)
			fresh = append(fresh, c)
		}
		c.retain()
		s.clusters[spec.Name] = c
	}
	for _, c := range previous {
		c.release()
	}
	s.fallback = s.clusters[topology.DefaultCluster]
	for _, c := range fresh {
		c.start()
	}
	return s, nil
}

// retainClusters holds every cluster of a Set that is still serving, so a
// Set built from it can adopt them before the old Set lets go.
func (s *Set) retainClusters() map[string]*cluster {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return nil
	}
	held := make(map[string]*cluster, len(s.clusters))
	for name, c := range s.clusters {
		c.retain()
		held[name] = c
	}
	return held
}

// Topology returns the topology the Set serves. The caller must not modify
// it.
func (s *Set) Topology() Topology { return s.topology }

// Warm waits until every health-checked endpoint has finished its first
// check, which is when Envoy finishes warming a cluster. Until then such an
// endpoint counts as unhealthy, so a Set that serves early balances over all
// endpoints through the panic threshold. Warm returns ctx's error if it ends
// first.
func (s *Set) Warm(ctx context.Context) error {
	for _, c := range s.clusters {
		if err := c.warm(ctx); err != nil {
			return err
		}
	}
	return nil
}

// Do sends req to the cluster its route key selects. It returns once the
// response is ready to commit; the caller must close the response body,
// which ends the call. When no attempt produced a response, Do answers as
// Envoy does, with a local reply that Response.Local marks. It returns an
// *Error only when there is nothing to answer: the request is invalid or
// unroutable, the caller went away, or the Set is closed.
func (s *Set) Do(ctx context.Context, req *Request) (*Response, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	if !s.enter() {
		return nil, &Error{Kind: KindClosed, Err: errSetClosed}
	}
	c, defaultRoute := s.route(req.RouteKey)
	if c == nil {
		s.exit()
		return nil, &Error{Kind: KindNoRoute, Err: fmt.Errorf("no cluster serves route key %q", req.RouteKey)}
	}
	policy := s.policy(req, c)
	callCtx, endCall := s.callContext(ctx, policy.Timeouts.Total)
	cl := &call{req: req, cluster: c, defaultRoute: defaultRoute, policy: policy, rnd: s.rnd, clock: s.clock}
	done := func() {
		endCall()
		s.exit()
	}
	resp, failure := cl.run(callCtx)
	if failure == nil {
		cl.served.onEnd = done
		metrics.RecordUpstreamRequest(c.spec.Name, statusClass(resp.StatusCode))
		return resp, nil
	}
	done()
	if !failure.answered() {
		return nil, failure
	}
	metrics.RecordUpstreamRequest(c.spec.Name, string(failure.Kind))
	return cl.localReply(failure), nil
}

// route finds the cluster for a route key. Like the template's catch-all
// route, a key no cluster serves falls through to the default route.
func (s *Set) route(key string) (*cluster, bool) {
	if c, ok := s.clusters[key]; ok && key != "" {
		return c, false
	}
	return s.fallback, true
}

// policy layers the call's policy: built-in defaults, then the listener's
// route defaults, the cluster's own policy, the decision's reliability
// override, and the caller's override.
func (s *Set) policy(req *Request, c *cluster) Policy {
	listener, ok := s.listeners[req.Listener]
	if !ok {
		listener = s.first
	}
	policy := listener.over(builtinPolicy())
	policy = c.spec.Policy.over(policy).merge(req.Reliability...)
	if req.Policy != nil {
		policy = req.Policy.over(policy)
	}
	return policy
}

// callContext bounds a call by its total timeout and by a forced Close.
func (s *Set) callContext(parent context.Context, total time.Duration) (context.Context, func()) {
	ctx, cancel := context.WithCancelCause(parent)
	stopAbort := context.AfterFunc(s.abort, func() { cancel(errSetClosed) })
	if !enabled(total) {
		return ctx, func() { stopAbort(); cancel(nil) }
	}
	timed, stopTimer := context.WithTimeoutCause(ctx, total, &timeoutCause{stage: StageTotal})
	return timed, func() { stopAbort(); stopTimer(); cancel(nil) }
}

func (s *Set) enter() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	s.inflight++
	return true
}

func (s *Set) exit() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inflight--
	if s.closed && s.inflight == 0 {
		close(s.drained)
	}
}

// Close stops the Set from accepting calls and waits until its in-flight
// calls end, streamed bodies included. If ctx ends first, the remaining
// calls are aborted. Either way the Set then releases its clusters; those a
// newer Set adopted stay open.
func (s *Set) Close(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	if s.inflight == 0 {
		close(s.drained)
	}
	s.mu.Unlock()
	var err error
	select {
	case <-s.drained:
	case <-ctx.Done():
		err = ctx.Err()
	}
	s.cancel()
	s.release()
	return err
}

func (s *Set) release() {
	s.mu.Lock()
	if s.released {
		s.mu.Unlock()
		return
	}
	s.released = true
	s.mu.Unlock()
	for _, c := range s.clusters {
		c.release()
	}
}

func (t *Topology) validate() error {
	names := make(map[string]bool, len(t.Clusters))
	for i := range t.Clusters {
		spec := &t.Clusters[i]
		if spec.Name == "" || names[spec.Name] {
			return fmt.Errorf("cluster %d: name %q is empty or duplicated", i, spec.Name)
		}
		names[spec.Name] = true
		if err := spec.validate(); err != nil {
			return fmt.Errorf("cluster %s: %w", spec.Name, err)
		}
	}
	if t.DefaultCluster != "" && !names[t.DefaultCluster] {
		return fmt.Errorf("default cluster %q does not exist", t.DefaultCluster)
	}
	return nil
}

func (c *ClusterSpec) validate() error {
	if len(c.Endpoints) == 0 {
		return fmt.Errorf("has no endpoints")
	}
	if c.LBPolicy != LBRoundRobin && c.LBPolicy != LBLeastRequest {
		return fmt.Errorf("unknown lb policy %q", c.LBPolicy)
	}
	for _, h := range c.RouteHeaders {
		if !httpguts.ValidHeaderFieldName(h.Name) || !httpguts.ValidHeaderFieldValue(h.Value) {
			return fmt.Errorf("route header %q is not a valid HTTP header", h.Name)
		}
	}
	if o := c.Outlier; o != nil && (o.Consecutive5xx < 0 || o.MaxEjectionPercent < 0 || o.MaxEjectionPercent > 100) {
		return fmt.Errorf("outlier detection needs consecutive_5xx >= 0 and max_ejection_percent in 0..100")
	}
	if h := c.HealthCheck; h != nil && (len(h.Path) == 0 || h.Path[0] != '/') {
		return fmt.Errorf("health check path %q must start with /", h.Path)
	}
	for _, ep := range c.Endpoints {
		wantTLS := ep.Scheme == schemeHTTPS
		switch {
		case ep.Scheme != schemeHTTP && !wantTLS:
			return fmt.Errorf("endpoint %s: unsupported scheme %q", ep.Name, ep.Scheme)
		case wantTLS != (c.TLS != nil):
			return fmt.Errorf("endpoint %s: scheme %q does not match the cluster's TLS setting", ep.Name, ep.Scheme)
		case ep.Host == "" || ep.Port < 1 || ep.Port > 65535 || ep.Weight < 1:
			return fmt.Errorf("endpoint %s: invalid host, port or weight", ep.Name)
		}
	}
	return nil
}
