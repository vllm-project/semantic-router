package upstream

import (
	"errors"
	"fmt"
	"net/netip"
	"net/url"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// defaultListenerName is the listener the CLI renders when a config declares
// none.
const defaultListenerName = "listener_0"

// Build compiles cfg and builds a Set from it.
func Build(cfg *config.RouterConfig, opts Options) (*Set, error) {
	topology, err := Compile(cfg)
	if err != nil {
		return nil, err
	}
	return New(topology, opts)
}

// Compile derives the upstream topology from the router configuration with
// the rules the CLI uses to render the Envoy template from the same
// document: one cluster per provider model alias that has a backend, in
// authored order, the first of them serving the default route.
func Compile(cfg *config.RouterConfig) (Topology, error) {
	if cfg == nil {
		return Topology{}, errors.New("router config is nil")
	}
	listeners, err := compileListeners(cfg.Listeners)
	if err != nil {
		return Topology{}, err
	}
	topology := Topology{Listeners: listeners}
	byModel := endpointsByModel(cfg.VLLMEndpoints)
	for _, alias := range clusterOrder(cfg, byModel) {
		spec, err := compileCluster(cfg, alias, byModel[alias])
		if err != nil {
			return Topology{}, err
		}
		topology.Clusters = append(topology.Clusters, spec)
	}
	if len(topology.Clusters) > 0 {
		topology.DefaultCluster = topology.Clusters[0].Name
	}
	return topology, nil
}

func compileListeners(listeners []config.Listener) ([]ListenerSpec, error) {
	if len(listeners) == 0 {
		return []ListenerSpec{{Name: defaultListenerName, Timeouts: routeTimeouts(defaultRouteTimeout)}}, nil
	}
	specs := make([]ListenerSpec, 0, len(listeners))
	for _, listener := range listeners {
		timeout := defaultRouteTimeout
		if raw := strings.TrimSpace(listener.Timeout); raw != "" {
			parsed, err := time.ParseDuration(raw)
			if err != nil || parsed < 0 {
				return nil, fmt.Errorf("listeners[%s].timeout %q is not a valid duration", listener.Name, raw)
			}
			timeout = parsed
		}
		specs = append(specs, ListenerSpec{Name: listener.Name, Timeouts: routeTimeouts(timeout)})
	}
	return specs, nil
}

// routeTimeouts maps a listener timeout onto the two Envoy timeouts the
// template derives from it: the route timeout and the stream idle timeout.
// Zero disables both, as it does in Envoy.
func routeTimeouts(timeout time.Duration) Timeouts {
	if timeout == 0 {
		timeout = NoTimeout
	}
	return Timeouts{Total: timeout, Idle: timeout}
}

func endpointsByModel(endpoints []config.VLLMEndpoint) map[string][]config.VLLMEndpoint {
	byModel := make(map[string][]config.VLLMEndpoint)
	for _, endpoint := range endpoints {
		if endpoint.Model != "" {
			byModel[endpoint.Model] = append(byModel[endpoint.Model], endpoint)
		}
	}
	return byModel
}

// clusterOrder lists the aliases that have backends: authored order first,
// then any alias the config did not order, by name.
func clusterOrder(cfg *config.RouterConfig, byModel map[string][]config.VLLMEndpoint) []string {
	order := make([]string, 0, len(byModel))
	seen := make(map[string]bool, len(byModel))
	for _, alias := range cfg.ProviderModelOrder {
		if len(byModel[alias]) > 0 && !seen[alias] {
			order = append(order, alias)
			seen[alias] = true
		}
	}
	var rest []string
	for alias := range byModel {
		if !seen[alias] {
			rest = append(rest, alias)
		}
	}
	sort.Strings(rest)
	return append(order, rest...)
}

func compileCluster(cfg *config.RouterConfig, alias string, endpoints []config.VLLMEndpoint) (ClusterSpec, error) {
	reliability := cfg.ModelConfig[alias].Reliability
	spec := ClusterSpec{Name: alias, LBPolicy: LBRoundRobin}
	if reliability.LBPolicy == config.ProviderLBPolicyLeastRequest {
		spec.LBPolicy = LBLeastRequest
	}
	for i, endpoint := range endpoints {
		ep, err := compileEndpoint(alias, i, endpoint)
		if err != nil {
			return ClusterSpec{}, err
		}
		spec.Endpoints = append(spec.Endpoints, ep)
	}
	// One endpoint behind a DNS name is a LOGICAL_DNS cluster in the
	// template, which resolves IPv4 addresses only.
	if len(spec.Endpoints) == 1 && !isIPLiteral(spec.Endpoints[0].Host) {
		spec.Endpoints[0].IPv4Only = true
	}
	// Backend pools are homogeneous (the config loader rejects mixed
	// schemes, base paths, TLS names and headers), so the first endpoint
	// speaks for the cluster, as it does in the template.
	first := spec.Endpoints[0]
	if first.Scheme == schemeHTTPS {
		spec.TLS = &TLSSpec{ServerName: first.Host}
	}
	profile := cfg.ProviderProfiles[endpoints[0].ProviderProfileName]
	prefix, err := basePath(profile.BaseURL)
	if err != nil {
		return ClusterSpec{}, fmt.Errorf("providers.models[%s].backend_refs[0]: %w", alias, err)
	}
	spec.PathPrefix = prefix
	spec.RouteHeaders = routeHeaders(profile.ExtraHeaders)
	if err := compileHealth(&spec, reliability); err != nil {
		return ClusterSpec{}, fmt.Errorf("providers.models[%s].reliability: %w", alias, err)
	}
	if err := compilePolicy(&spec, reliability); err != nil {
		return ClusterSpec{}, fmt.Errorf("providers.models[%s].reliability: %w", alias, err)
	}
	return spec, nil
}

// compilePolicy maps the reliability block's timeouts and retry policy. A
// retry policy exists only with retries, as the template renders it.
func compilePolicy(spec *ClusterSpec, r config.ProviderReliability) error {
	var timeouts Timeouts
	var err error
	for _, field := range []struct {
		name   string
		raw    string
		out    *time.Duration
		zeroOK bool
	}{
		{"connect_timeout", r.ConnectTimeout, &timeouts.Connect, false},
		{"total_timeout", r.TotalTimeout, &timeouts.Total, true},
		{"idle_timeout", r.IdleTimeout, &timeouts.Idle, true},
		{"per_try_timeout", r.PerTryTimeout, &timeouts.PerTry, false},
		{"first_byte_timeout", r.FirstByteTimeout, &timeouts.FirstByte, false},
	} {
		if *field.out, err = timeoutField(field.name, field.raw, field.zeroOK); err != nil {
			return err
		}
	}
	spec.Policy.Timeouts = timeouts
	if r.RetryBudgetPercent > 0 || r.RetryBudgetMinConcurrency > 0 {
		spec.Breakers.RetryBudget = &RetryBudget{Percent: r.RetryBudgetPercent, MinConcurrency: r.RetryBudgetMinConcurrency}
	}
	// The Envoy template renders a route retry policy, even with no retries,
	// when a per-try timeout is set; a decision's retry override merges into it.
	if r.RetryCount == 0 && strings.TrimSpace(r.PerTryTimeout) == "" {
		return nil
	}
	retryOn := r.RetryOn
	if strings.TrimSpace(retryOn) == "" {
		retryOn = config.DefaultProviderRetryOn
	}
	retry := &RetryPolicy{
		NumRetries:           r.RetryCount,
		On:                   ParseRetryOn(retryOn),
		RetriableStatusCodes: slices.Clone(r.RetriableStatusCodes),
	}
	for _, field := range []struct {
		name string
		raw  string
		out  *time.Duration
	}{
		{"retry_back_off_base", r.RetryBackOffBase, &retry.BackOffBase},
		{"retry_back_off_max", r.RetryBackOffMax, &retry.BackOffMax},
		{"retry_after_max", r.RetryAfterMax, &retry.RetryAfterMax},
	} {
		if *field.out, err = optionalDuration(field.name, field.raw); err != nil {
			return err
		}
	}
	spec.Policy.Retry = retry
	return nil
}

// timeoutField parses a reliability timeout. An explicit 0 disables the
// timeout where Envoy allows that, which NoTimeout records.
func timeoutField(field, raw string, zeroOK bool) (time.Duration, error) {
	if zeroOK && strings.TrimSpace(raw) != "" {
		if d, err := time.ParseDuration(strings.TrimSpace(raw)); err == nil && d == 0 {
			return NoTimeout, nil
		}
	}
	return optionalDuration(field, raw)
}

// raisedMaxRequests is the max_requests the template sets once retries or
// outlier detection are enabled.
const raisedMaxRequests = 4096

// compileHealth maps the reliability block onto circuit breakers, outlier
// detection and active health checks with the template's conditions: the
// outlier block needs consecutive_5xx and more than one endpoint, and health
// checks need a path.
func compileHealth(spec *ClusterSpec, reliability config.ProviderReliability) error {
	if reliability.RetryCount > 0 || reliability.Consecutive5xx > 0 {
		spec.Breakers.MaxRequests = raisedMaxRequests
	}
	if reliability.Consecutive5xx > 0 && len(spec.Endpoints) > 1 {
		base, err := optionalDuration("base_ejection_time", reliability.BaseEjectionTime)
		if err != nil {
			return err
		}
		spec.Outlier = &OutlierSpec{
			Consecutive5xx:     reliability.Consecutive5xx,
			BaseEjectionTime:   base,
			MaxEjectionPercent: reliability.MaxEjectionPercent,
		}
	}
	if reliability.HealthCheckPath == "" {
		return nil
	}
	interval, err := optionalDuration("health_check_interval", reliability.HealthCheckInterval)
	if err != nil {
		return err
	}
	timeout, err := optionalDuration("health_check_timeout", reliability.HealthCheckTimeout)
	if err != nil {
		return err
	}
	spec.HealthCheck = &HealthCheckSpec{Path: reliability.HealthCheckPath, Interval: interval, Timeout: timeout}
	return nil
}

// optionalDuration parses a reliability duration; empty leaves the default.
func optionalDuration(field, raw string) (time.Duration, error) {
	if raw = strings.TrimSpace(raw); raw == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("%s %q is not a positive duration", field, raw)
	}
	return d, nil
}

func compileEndpoint(alias string, index int, endpoint config.VLLMEndpoint) (EndpointSpec, error) {
	scheme := strings.ToLower(strings.TrimSpace(endpoint.Protocol))
	if scheme == "" {
		scheme = schemeHTTP
	}
	where := fmt.Sprintf("providers.models[%s].backend_refs[%d]", alias, index)
	switch {
	case scheme != schemeHTTP && scheme != schemeHTTPS:
		return EndpointSpec{}, fmt.Errorf("%s scheme %q is unsupported", where, endpoint.Protocol)
	case endpoint.Address == "":
		return EndpointSpec{}, fmt.Errorf("%s has no host", where)
	case endpoint.Port < 1 || endpoint.Port > 65535:
		return EndpointSpec{}, fmt.Errorf("%s has an invalid port %d; expected 1..65535", where, endpoint.Port)
	case scheme == schemeHTTPS && isIPLiteral(endpoint.Address):
		return EndpointSpec{}, fmt.Errorf("%s HTTPS endpoint must use a DNS hostname so its certificate identity can be verified", where)
	}
	return EndpointSpec{
		Name:   endpoint.Name,
		Scheme: scheme,
		Host:   endpoint.Address,
		Port:   endpoint.Port,
		Weight: max(endpoint.Weight, 1),
	}, nil
}

// basePath is the path of a provider base URL without its trailing slash.
func basePath(baseURL string) (string, error) {
	if baseURL == "" {
		return "", nil
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid base URL: %w", err)
	}
	return strings.TrimRight(parsed.EscapedPath(), "/"), nil
}

// routeHeaders orders the endpoint's extra headers by name, skipping empty
// names, as the template data does.
func routeHeaders(extra map[string]string) []Header {
	headers := make([]Header, 0, len(extra))
	for name, value := range extra {
		if name != "" {
			headers = append(headers, Header{Name: name, Value: value})
		}
	}
	sort.Slice(headers, func(i, j int) bool { return headers[i].Name < headers[j].Name })
	if len(headers) == 0 {
		return nil
	}
	return headers
}

func isIPLiteral(host string) bool {
	_, err := netip.ParseAddr(host)
	return err == nil
}
