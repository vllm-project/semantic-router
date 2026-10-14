package upstream

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const compileFixtureListeners = `
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
    timeout: 120s
  - name: internal
    address: 127.0.0.1
    port: 8900
    timeout: 0s
`

const compileFixture = "version: v0.3\n" + compileFixtureListeners + compileFixtureProviders

const compileFixtureProviders = `
providers:
  defaults:
    model: zeta
  models:
    - name: zeta
      provider_model_id: zeta-model
      api_format: openai
      reliability:
        lb_policy: least_request
      backend_refs:
        - name: a
          provider: vllm
          endpoint: 10.0.0.1:8000
          weight: 3
        - name: b
          provider: vllm
          endpoint: 10.0.0.2:8000
    - name: alpha
      provider_model_id: alpha-model
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: https://api.example.test/compatible-mode/v1/
          extra_headers:
            X-Tenant: one
            X-Region: west
routing: {}
`

func compileYAML(t *testing.T, yaml string) Topology {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	topology, err := Compile(cfg)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return topology
}

func TestCompileFollowsAuthoredOrderForTheDefaultRoute(t *testing.T) {
	topology := compileYAML(t, compileFixture)
	var names []string
	for _, c := range topology.Clusters {
		names = append(names, c.Name)
	}
	if !reflect.DeepEqual(names, []string{"zeta", "alpha"}) || topology.DefaultCluster != "zeta" {
		t.Fatalf("clusters = %v, default = %q; want authored order with zeta as default", names, topology.DefaultCluster)
	}
}

func TestCompileMapsEndpointsAndRouteBehavior(t *testing.T) {
	topology := compileYAML(t, compileFixture)
	zeta := topology.cluster("zeta")
	wantZeta := []EndpointSpec{
		{Name: "zeta_a", Scheme: "http", Host: "10.0.0.1", Port: 8000, Weight: 3},
		{Name: "zeta_b", Scheme: "http", Host: "10.0.0.2", Port: 8000, Weight: 1},
	}
	if !reflect.DeepEqual(zeta.Endpoints, wantZeta) || zeta.LBPolicy != LBLeastRequest || zeta.TLS != nil {
		t.Fatalf("zeta = %+v", zeta)
	}
	if zeta.PathPrefix != "" || zeta.RouteHeaders != nil {
		t.Fatalf("zeta route behavior = %q %v, want none", zeta.PathPrefix, zeta.RouteHeaders)
	}

	alpha := topology.cluster("alpha")
	wantAlpha := []EndpointSpec{{
		Name: "alpha_primary", Scheme: "https", Host: "api.example.test", Port: 443, Weight: 1, IPv4Only: true,
	}}
	if !reflect.DeepEqual(alpha.Endpoints, wantAlpha) || alpha.LBPolicy != LBRoundRobin {
		t.Fatalf("alpha endpoints = %+v policy = %s", alpha.Endpoints, alpha.LBPolicy)
	}
	if alpha.TLS == nil || alpha.TLS.ServerName != "api.example.test" {
		t.Fatalf("alpha TLS = %+v, want SNI api.example.test", alpha.TLS)
	}
	if alpha.PathPrefix != "/compatible-mode/v1" {
		t.Fatalf("alpha path prefix = %q", alpha.PathPrefix)
	}
	wantHeaders := []Header{{Name: "X-Region", Value: "west"}, {Name: "X-Tenant", Value: "one"}}
	if !reflect.DeepEqual(alpha.RouteHeaders, wantHeaders) {
		t.Fatalf("alpha route headers = %v, want %v", alpha.RouteHeaders, wantHeaders)
	}
}

func TestCompileListenerTimeoutsBecomeRouteDefaults(t *testing.T) {
	topology := compileYAML(t, compileFixture)
	want := []ListenerSpec{
		{Name: "public", Timeouts: Timeouts{Total: 120 * time.Second, Idle: 120 * time.Second}},
		{Name: "internal", Timeouts: Timeouts{Total: NoTimeout, Idle: NoTimeout}},
	}
	if !reflect.DeepEqual(topology.Listeners, want) {
		t.Fatalf("listeners = %+v, want %+v", topology.Listeners, want)
	}

	defaults := compileYAML(t, "version: v0.3\n"+compileFixtureProviders).Listeners
	wantDefault := []ListenerSpec{{Name: "listener_0", Timeouts: Timeouts{Total: 300 * time.Second, Idle: 300 * time.Second}}}
	if !reflect.DeepEqual(defaults, wantDefault) {
		t.Fatalf("default listeners = %+v, want %+v", defaults, wantDefault)
	}
}

const reliabilityFixture = `
version: v0.3
providers:
  models:
    - name: pooled
      provider_model_id: pooled-model
      api_format: openai
      reliability:
        retry_count: 2
        retry_on: connect-failure
        consecutive_5xx: 3
        base_ejection_time: 45s
        max_ejection_percent: 30
        health_check_path: /health
        health_check_interval: 5s
        health_check_timeout: 1s
      backend_refs:
        - {name: a, provider: vllm, endpoint: 10.0.0.1:8000}
        - {name: b, provider: vllm, endpoint: 10.0.0.2:8000}
    - name: single
      provider_model_id: single-model
      api_format: openai
      reliability:
        consecutive_5xx: 3
      backend_refs:
        - {name: a, provider: vllm, endpoint: 10.0.0.3:8000}
    - name: plain
      provider_model_id: plain-model
      api_format: openai
      backend_refs:
        - {name: a, provider: vllm, endpoint: 10.0.0.4:8000}
        - {name: b, provider: vllm, endpoint: 10.0.0.5:8000}
routing: {}
`

func TestCompileMapsReliabilityWithTheTemplateConditions(t *testing.T) {
	topology := compileYAML(t, reliabilityFixture)
	pooled := topology.cluster("pooled")
	if pooled.Breakers != (Breakers{MaxRequests: 4096}) {
		t.Fatalf("pooled breakers = %+v, want max_requests raised to 4096", pooled.Breakers)
	}
	wantOutlier := &OutlierSpec{Consecutive5xx: 3, BaseEjectionTime: 45 * time.Second, MaxEjectionPercent: 30}
	if !reflect.DeepEqual(pooled.Outlier, wantOutlier) {
		t.Fatalf("pooled outlier = %+v, want %+v", pooled.Outlier, wantOutlier)
	}
	wantHealth := &HealthCheckSpec{Path: "/health", Interval: 5 * time.Second, Timeout: time.Second}
	if !reflect.DeepEqual(pooled.HealthCheck, wantHealth) {
		t.Fatalf("pooled health check = %+v, want %+v", pooled.HealthCheck, wantHealth)
	}
	// The template renders outlier detection only for more than one endpoint.
	single := topology.cluster("single")
	if single.Outlier != nil || single.Breakers.MaxRequests != 4096 {
		t.Fatalf("single = %+v", single)
	}
	plain := topology.cluster("plain")
	if plain.Outlier != nil || plain.HealthCheck != nil || plain.Breakers != (Breakers{}) {
		t.Fatalf("plain = %+v, want Envoy's defaults only", plain)
	}
}

func TestWithoutActiveHealthChecksKeepsPassiveEjection(t *testing.T) {
	topology := compileYAML(t, reliabilityFixture)
	passive := topology.WithoutActiveHealthChecks()
	pooled := passive.cluster("pooled")
	if pooled.HealthCheck != nil || pooled.Outlier == nil || pooled.Breakers.MaxRequests != 4096 {
		t.Fatalf("pooled = %+v, want outlier detection and breakers without health checks", pooled)
	}
	if topology.cluster("pooled").HealthCheck == nil {
		t.Fatal("the source topology lost its health checks")
	}
	if passive.DefaultCluster != topology.DefaultCluster || len(passive.Clusters) != len(topology.Clusters) {
		t.Fatalf("routes changed: %+v", passive)
	}
}

func TestReliabilityDefaultsAreTheTemplatesAndEnvoys(t *testing.T) {
	outlier := (OutlierSpec{Consecutive5xx: 5}).withDefaults()
	wantOutlier := OutlierSpec{
		Consecutive5xx: 5, Interval: 10 * time.Second, BaseEjectionTime: 30 * time.Second,
		MaxEjectionTime: 300 * time.Second, MaxEjectionPercent: 50,
		SuccessRateMinimumHosts: 5, SuccessRateRequestVolume: 100, SuccessRateStdevFactor: 1.9,
	}
	if outlier != wantOutlier {
		t.Fatalf("outlier defaults = %+v, want %+v", outlier, wantOutlier)
	}
	if long := (OutlierSpec{BaseEjectionTime: 10 * time.Minute}).withDefaults(); long.MaxEjectionTime != 10*time.Minute {
		t.Fatalf("max ejection time = %v, want at least the base", long.MaxEjectionTime)
	}
	health := (HealthCheckSpec{Path: "/health"}).withDefaults()
	wantHealth := HealthCheckSpec{
		Path: "/health", Interval: 10 * time.Second, NoTrafficInterval: 60 * time.Second, Timeout: 2 * time.Second,
		UnhealthyThreshold: 3, HealthyThreshold: 1,
	}
	if health != wantHealth {
		t.Fatalf("health defaults = %+v, want %+v", health, wantHealth)
	}
}

func TestCompileRejectsWhatEnvoyRenderingRejects(t *testing.T) {
	tests := []struct {
		name     string
		endpoint config.VLLMEndpoint
		want     string
	}{
		{
			name:     "https to an IP literal",
			endpoint: config.VLLMEndpoint{Name: "e", Address: "10.0.0.1", Port: 443, Protocol: "https", Model: "m"},
			want:     "DNS hostname",
		},
		{
			name:     "unsupported scheme",
			endpoint: config.VLLMEndpoint{Name: "e", Address: "backend", Port: 80, Protocol: "ftp", Model: "m"},
			want:     "unsupported",
		},
		{
			name:     "invalid port",
			endpoint: config.VLLMEndpoint{Name: "e", Address: "backend", Port: 0, Model: "m"},
			want:     "invalid port",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Compile(&config.RouterConfig{BackendModels: config.BackendModels{
				VLLMEndpoints: []config.VLLMEndpoint{tt.endpoint},
			}})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Compile error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestCompileWithoutAuthoredOrderSortsAliases(t *testing.T) {
	topology, err := Compile(&config.RouterConfig{BackendModels: config.BackendModels{
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "m2", Address: "10.0.0.2", Port: 80, Model: "zz"},
			{Name: "m1", Address: "10.0.0.1", Port: 80, Model: "aa"},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if topology.DefaultCluster != "aa" || len(topology.Clusters) != 2 {
		t.Fatalf("topology = %+v", topology)
	}
}

func TestCompileWithoutBackendsHasNoDefaultRoute(t *testing.T) {
	topology, err := Compile(&config.RouterConfig{})
	if err != nil {
		t.Fatal(err)
	}
	set, err := New(topology, Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = set.Do(t.Context(), &Request{Method: "POST", Path: "/v1/chat/completions"})
	if KindOf(err) != KindNoRoute {
		t.Fatalf("Do error = %v, want %s", err, KindNoRoute)
	}
}
