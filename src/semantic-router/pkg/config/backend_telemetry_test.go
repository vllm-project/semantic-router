package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

func telemetryDocument(backendRefs string) string {
	return `version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
` + backendRefs + `
routing:
  modelCards:
    - name: qwen3-8b
`
}

const telemetryBackendRefs = `        - name: gpu-a
          endpoint: 10.0.0.1:8000
          provider: vllm
          engine: vllm
        - name: gpu-b
          endpoint: 10.0.0.2:8000
          provider: vllm
          engine: tensorrt-llm
          metrics:
            port: 8002
            path: /v2/metrics
        - name: gpu-c
          endpoint: 10.0.0.3:8000
          provider: vllm`

func endpointByRef(t *testing.T, cfg *RouterConfig, ref string) VLLMEndpoint {
	t.Helper()
	for _, endpoint := range cfg.VLLMEndpoints {
		if endpoint.BackendRefName() == ref {
			return endpoint
		}
	}
	t.Fatalf("no runtime endpoint for backend ref %q in %+v", ref, cfg.VLLMEndpoints)
	return VLLMEndpoint{}
}

func TestBackendRefTelemetryReachesRuntimeEndpoints(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(telemetryDocument(telemetryBackendRefs)))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		ref        string
		identity   backendtelemetry.Identity
		metricsURL string
	}{
		{"gpu-a", backendtelemetry.Identity{Model: "qwen3-8b", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM}, "http://10.0.0.1:8000/metrics"},
		{"gpu-b", backendtelemetry.Identity{Model: "qwen3-8b", BackendRef: "gpu-b", Engine: backendtelemetry.EngineTensorRTLLM}, "http://10.0.0.2:8002/v2/metrics"},
	}
	for _, tc := range cases {
		endpoint := endpointByRef(t, cfg, tc.ref)
		if !endpoint.Telemetry.Enabled() {
			t.Fatalf("%s: telemetry disabled", tc.ref)
		}
		if got := endpoint.TelemetryIdentity(); got != tc.identity {
			t.Fatalf("%s: identity = %+v, want %+v", tc.ref, got, tc.identity)
		}
		if got := endpoint.MetricsURL(); got != tc.metricsURL {
			t.Fatalf("%s: metrics URL = %q, want %q", tc.ref, got, tc.metricsURL)
		}
	}
	if endpointByRef(t, cfg, "gpu-c").Telemetry.Enabled() {
		t.Fatal("a backend without an engine reports telemetry")
	}
}

func TestBackendRefTelemetryRoundTripsThroughCanonicalExport(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(telemetryDocument(telemetryBackendRefs)))
	if err != nil {
		t.Fatal(err)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	refs := map[string]CanonicalBackendRef{}
	for _, model := range exported.Providers.Models {
		for _, ref := range model.BackendRefs {
			refs[ref.Name] = ref
		}
	}
	if got := refs["gpu-a"]; got.Engine != "vllm" || got.Metrics != nil {
		t.Fatalf("gpu-a exported as engine %q metrics %+v", got.Engine, got.Metrics)
	}
	if got := refs["gpu-b"]; got.Engine != "tensorrt-llm" || got.Metrics == nil ||
		*got.Metrics != (CanonicalBackendMetrics{Port: 8002, Path: "/v2/metrics"}) {
		t.Fatalf("gpu-b exported as engine %q metrics %+v", got.Engine, got.Metrics)
	}
	if got := refs["gpu-c"]; got.Engine != "" || got.Metrics != nil {
		t.Fatalf("gpu-c exported telemetry it never declared: %+v", got)
	}

	document, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatal(err)
	}
	reparsed, err := ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("exported config does not load: %v", err)
	}
	if got := endpointByRef(t, reparsed, "gpu-b").MetricsURL(); got != "http://10.0.0.2:8002/v2/metrics" {
		t.Fatalf("round-tripped metrics URL = %q", got)
	}
}

func TestBackendRefTelemetryValidation(t *testing.T) {
	cases := []struct {
		name string
		ref  string
		want string
	}{
		{"unknown engine", "          engine: triton", "backend_refs[0].engine: unknown engine \"triton\""},
		{"metrics without engine", "          metrics:\n            port: 9000", "backend_refs[0].metrics requires engine"},
		{"bad port", "          engine: vllm\n          metrics:\n            port: 70000", "metrics.port 70000"},
		{"relative path", "          engine: vllm\n          metrics:\n            path: metrics", "metrics.path \"metrics\""},
		{"path with query", "          engine: vllm\n          metrics:\n            path: /metrics?x=1", "metrics.path"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			refs := "        - name: gpu-a\n          endpoint: 10.0.0.1:8000\n          provider: vllm\n" + tc.ref
			_, err := ParseYAMLBytes([]byte(telemetryDocument(refs)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseYAMLBytes() = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestEndpointMetricsURLBracketsIPv6(t *testing.T) {
	endpoint := VLLMEndpoint{Address: "fd00::1", Port: 8000, Telemetry: EndpointTelemetry{Engine: "vllm"}}
	if got := endpoint.MetricsURL(); got != "http://[fd00::1]:8000/metrics" {
		t.Fatalf("MetricsURL() = %q", got)
	}
}
