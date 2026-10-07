package upstream

import (
	"net"
	"net/netip"
	"reflect"
	"strconv"
	"time"
)

// LBPolicy selects how a cluster spreads requests over its endpoints.
type LBPolicy string

const (
	// LBRoundRobin rotates over the endpoints, following their weights when
	// they differ.
	LBRoundRobin LBPolicy = "round_robin"
	// LBLeastRequest prefers the endpoint with the fewest active requests.
	LBLeastRequest LBPolicy = "least_request"
)

// Topology is the compiled, immutable description a Set serves: every
// cluster, the default route, and the route defaults of each listener.
// Compile derives it from the router configuration; a configuration snapshot
// can build it directly.
type Topology struct {
	// Clusters are in declaration order.
	Clusters []ClusterSpec
	// DefaultCluster names the cluster that serves a request without a route
	// key, or with a key no cluster matches. It is empty when there are no
	// clusters.
	DefaultCluster string
	// Listeners carry the route defaults of each frontend listener. A request
	// that names no listener gets the first one.
	Listeners []ListenerSpec
}

// WithoutActiveHealthChecks returns a copy of the topology whose clusters
// run no active health checks. Passive outlier detection stays.
func (t Topology) WithoutActiveHealthChecks() Topology {
	clusters := make([]ClusterSpec, len(t.Clusters))
	for i, spec := range t.Clusters {
		spec.HealthCheck = nil
		clusters[i] = spec
	}
	t.Clusters = clusters
	return t
}

// ListenerSpec holds the route defaults one listener applies to every
// request it accepts, below the cluster's own policy.
type ListenerSpec struct {
	Name     string
	Timeouts Timeouts
}

// ClusterSpec describes one cluster: a provider model alias, its endpoints,
// and the route behavior applied to requests sent to it.
type ClusterSpec struct {
	// Name is the route key that selects the cluster.
	Name      string
	Endpoints []EndpointSpec
	LBPolicy  LBPolicy
	// TLS is nil when the endpoints use plain HTTP.
	TLS *TLSSpec
	// PathPrefix is the path of the endpoints' base URL. Only the default
	// route uses it, to rewrite a /v1 request path onto the provider path;
	// routed requests already carry the complete provider path.
	PathPrefix string
	// RouteHeaders are set on every request, replacing any value present.
	RouteHeaders []Header
	// Policy is the cluster's own timeouts and retry policy, between the
	// listener's route defaults and a call's override.
	Policy Policy
	// Breakers cap the cluster's concurrent work; zero fields take Envoy's
	// defaults.
	Breakers Breakers
	// Outlier enables passive ejection; nil disables it.
	Outlier *OutlierSpec
	// HealthCheck enables active health checks; nil disables them.
	HealthCheck *HealthCheckSpec
}

// Breakers are a cluster's circuit-breaker thresholds. Zero fields take
// Envoy's defaults.
type Breakers struct {
	// MaxConnections bounds the requests holding an upstream connection; an
	// HTTP/1.1 connection carries one request at a time.
	MaxConnections int
	// MaxPendingRequests bounds the requests waiting for a connection.
	MaxPendingRequests int
	// MaxRequests bounds the requests in flight.
	MaxRequests int
	// MaxRetries bounds the retries in flight, unless RetryBudget is set.
	MaxRetries int
	// RetryBudget replaces MaxRetries with a share of the requests in flight.
	RetryBudget *RetryBudget
}

// OutlierSpec configures passive ejection the way the Envoy template's
// outlier_detection block does, with Envoy's defaults for what it omits.
type OutlierSpec struct {
	// Consecutive5xx failures eject an endpoint. Connection failures, resets
	// and timeouts count as failures.
	Consecutive5xx int
	// Interval is the period of the ejection sweep.
	Interval time.Duration
	// BaseEjectionTime is multiplied by how often the endpoint was ejected in
	// a row, up to MaxEjectionTime.
	BaseEjectionTime time.Duration
	MaxEjectionTime  time.Duration
	// MaxEjectionPercent caps the share of endpoints out at once; zero takes
	// the CLI default of 50.
	MaxEjectionPercent int
	// Success-rate ejection: once SuccessRateMinimumHosts endpoints each saw
	// SuccessRateRequestVolume requests in a sweep, an endpoint whose success
	// rate is below mean - SuccessRateStdevFactor * stdev is ejected.
	SuccessRateMinimumHosts  int
	SuccessRateRequestVolume int
	SuccessRateStdevFactor   float64
}

// HealthCheckSpec configures active HTTP health checks.
type HealthCheckSpec struct {
	// Path is requested with GET; only a 200 passes.
	Path     string
	Interval time.Duration
	// NoTrafficInterval replaces Interval until the cluster serves a request.
	NoTrafficInterval time.Duration
	Timeout           time.Duration
	// UnhealthyThreshold network failures in a row mark an endpoint unhealthy;
	// any status other than 200 does so at once.
	UnhealthyThreshold int
	// HealthyThreshold passes in a row mark it healthy again; the first check
	// after startup needs one.
	HealthyThreshold int
}

// EndpointSpec is one backend of a cluster.
type EndpointSpec struct {
	Name string
	// Scheme is "http" or "https".
	Scheme string
	// Host is a DNS name or an IP literal, without brackets.
	Host   string
	Port   int
	Weight int
	// IPv4Only restricts name resolution to IPv4 addresses.
	IPv4Only bool
}

// TLSSpec describes the upstream TLS session of a cluster.
type TLSSpec struct {
	// ServerName is sent as SNI and is the name the server certificate must
	// match.
	ServerName string
}

// Header is one header name and value.
type Header struct {
	Name  string
	Value string
}

// Address is the host and port to dial.
func (e EndpointSpec) Address() string {
	return net.JoinHostPort(e.Host, strconv.Itoa(e.Port))
}

// Authority is the Host header value for the endpoint: the host, bracketed
// when it is an IPv6 literal, with the port unless it is the scheme default.
func (e EndpointSpec) Authority() string {
	host := e.Host
	if addr, err := netip.ParseAddr(host); err == nil && addr.Is6() {
		host = "[" + host + "]"
	}
	if (e.Scheme == schemeHTTP && e.Port == 80) || (e.Scheme == schemeHTTPS && e.Port == 443) {
		return host
	}
	return host + ":" + strconv.Itoa(e.Port)
}

func (c *ClusterSpec) equal(other *ClusterSpec) bool {
	return reflect.DeepEqual(c, other)
}

func (t *Topology) cluster(name string) *ClusterSpec {
	for i := range t.Clusters {
		if t.Clusters[i].Name == name {
			return &t.Clusters[i]
		}
	}
	return nil
}

const (
	schemeHTTP  = "http"
	schemeHTTPS = "https"
)
