package tensorrtllm

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	// DefaultMetricsPath is the default Triton/TensorRT-LLM Prometheus path.
	DefaultMetricsPath = "/metrics"
	// DefaultTritonMetricsPort is Triton's default Prometheus metrics port. It
	// differs from the inference/HTTP port (8000), unlike vLLM which serves
	// metrics on the same port as inference.
	DefaultTritonMetricsPort = 8002
)

// TargetOptions controls how TensorRT-LLM metrics scrape targets are derived
// from the router config.
type TargetOptions struct {
	// MetricsPath overrides the default Prometheus path when a backend ref does
	// not set its own metrics_path.
	MetricsPath string
	// MetricsPort overrides the default Triton metrics port when a backend ref
	// does not set its own metrics_port.
	MetricsPort int
}

// TargetsFromRouterConfig builds TensorRT-LLM telemetry adapter targets from
// runtime backend refs. Only endpoints whose engine_kind is "tensorrt-llm" are
// collected. Each backend ref self-describes its metrics surface (option B):
// metrics_port/metrics_path on the ref take precedence over TargetOptions,
// which in turn fall back to Triton defaults.
func TargetsFromRouterConfig(cfg *config.RouterConfig, opts TargetOptions) []backend.AdapterTarget {
	if cfg == nil || len(cfg.ModelConfig) == 0 {
		return nil
	}

	modelNames := make([]string, 0, len(cfg.ModelConfig))
	for modelName := range cfg.ModelConfig {
		modelNames = append(modelNames, modelName)
	}
	sort.Strings(modelNames)

	targets := []backend.AdapterTarget{}
	seen := map[string]struct{}{}
	for _, modelName := range modelNames {
		for _, endpoint := range cfg.GetEndpointsForModel(modelName) {
			if !isTensorRTLLMEndpoint(endpoint) {
				continue
			}
			metricsEndpoint, ok := metricsEndpointForEndpoint(endpoint, opts)
			if !ok {
				continue
			}
			backendID := strings.TrimSpace(endpoint.BackendID)
			if backendID == "" {
				backendID = endpoint.Name
			}
			identity := backend.BackendIdentity{
				BackendID:  backendID,
				ModelName:  modelName,
				ReplicaID:  replicaIDForEndpoint(endpoint),
				Endpoint:   endpoint.Name,
				EngineKind: backend.EngineKindTensorRTLLM,
			}
			key := identity.Key()
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}

			targets = append(targets, backend.AdapterTarget{
				Identity:        identity,
				MetricsEndpoint: metricsEndpoint,
				Headers:         authHeaders(endpoint),
				Labels: map[string]string{
					"endpoint_name": endpoint.Name,
				},
			})
		}
	}
	return targets
}

func isTensorRTLLMEndpoint(endpoint config.VLLMEndpoint) bool {
	return strings.TrimSpace(strings.ToLower(endpoint.EngineKind)) == string(backend.EngineKindTensorRTLLM)
}

// replicaIDForEndpoint derives a stable per-target replica id. Triton is one
// process per metrics endpoint, so one target maps to exactly one replica
// (naive 1:1 identity — MIG / multi-instance fan-out is out of scope).
func replicaIDForEndpoint(endpoint config.VLLMEndpoint) string {
	if endpoint.Address != "" && endpoint.Port > 0 {
		return net.JoinHostPort(strings.Trim(endpoint.Address, "[]"), strconv.Itoa(endpoint.Port))
	}
	return endpoint.Name
}

func metricsEndpointForEndpoint(endpoint config.VLLMEndpoint, opts TargetOptions) (string, bool) {
	if endpoint.Address == "" {
		return "", false
	}
	protocol := strings.TrimSpace(strings.ToLower(endpoint.Protocol))
	if protocol == "" {
		protocol = "http"
	}
	if protocol != "http" && protocol != "https" {
		return "", false
	}

	metricsPort := endpoint.MetricsPort
	if metricsPort <= 0 {
		metricsPort = opts.MetricsPort
	}
	if metricsPort <= 0 {
		metricsPort = DefaultTritonMetricsPort
	}

	metricsPath := strings.TrimSpace(endpoint.MetricsPath)
	if metricsPath == "" {
		metricsPath = strings.TrimSpace(opts.MetricsPath)
	}
	if metricsPath == "" {
		metricsPath = DefaultMetricsPath
	}
	if !strings.HasPrefix(metricsPath, "/") {
		metricsPath = "/" + metricsPath
	}

	hostPort := net.JoinHostPort(strings.Trim(endpoint.Address, "[]"), strconv.Itoa(metricsPort))
	return fmt.Sprintf("%s://%s%s", protocol, hostPort, metricsPath), true
}

func authHeaders(endpoint config.VLLMEndpoint) map[string]string {
	apiKey := strings.TrimSpace(endpoint.APIKey)
	if apiKey == "" {
		return nil
	}
	return map[string]string{"Authorization": "Bearer " + apiKey}
}
