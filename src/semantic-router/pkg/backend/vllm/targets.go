package vllm

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const DefaultMetricsPath = "/metrics"

// TargetOptions controls how vLLM metrics scrape targets are derived from the
// router config.
type TargetOptions struct {
	MetricsPath       string
	IncludeLegacyVLLM bool
}

// TargetsFromRouterConfig builds vLLM telemetry adapter targets from runtime
// backend refs.
func TargetsFromRouterConfig(cfg *config.RouterConfig, opts TargetOptions) []backend.AdapterTarget {
	if cfg == nil || len(cfg.ModelConfig) == 0 {
		return nil
	}
	metricsPath := strings.TrimSpace(opts.MetricsPath)
	if metricsPath == "" {
		metricsPath = DefaultMetricsPath
	}
	if !strings.HasPrefix(metricsPath, "/") {
		metricsPath = "/" + metricsPath
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
			if !isCollectableVLLMEndpoint(endpoint, opts.IncludeLegacyVLLM) {
				continue
			}
			metricsEndpoint, ok := metricsEndpointForEndpoint(endpoint, metricsPath)
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
				Endpoint:   endpoint.Name,
				EngineKind: backend.EngineKindVLLM,
			}
			key := identity.Key()
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}

			target := backend.AdapterTarget{
				Identity:        identity,
				MetricsEndpoint: metricsEndpoint,
				Headers:         authHeaders(endpoint),
				Labels: map[string]string{
					"endpoint_name": endpoint.Name,
				},
			}
			targets = append(targets, target)
		}
	}
	return targets
}

func isCollectableVLLMEndpoint(endpoint config.VLLMEndpoint, includeLegacy bool) bool {
	engineKind := strings.TrimSpace(strings.ToLower(endpoint.EngineKind))
	if engineKind == string(backend.EngineKindVLLM) {
		return true
	}
	if engineKind != "" || !includeLegacy {
		return false
	}
	return endpoint.ProviderProfileName == "" && endpoint.Address != "" && endpoint.Port > 0
}

func metricsEndpointForEndpoint(endpoint config.VLLMEndpoint, metricsPath string) (string, bool) {
	if endpoint.Address == "" || endpoint.Port <= 0 {
		return "", false
	}
	protocol := strings.TrimSpace(strings.ToLower(endpoint.Protocol))
	if protocol == "" {
		protocol = "http"
	}
	if protocol != "http" && protocol != "https" {
		return "", false
	}
	hostPort := net.JoinHostPort(strings.Trim(endpoint.Address, "[]"), strconv.Itoa(endpoint.Port))
	return fmt.Sprintf("%s://%s%s", protocol, hostPort, metricsPath), true
}

func authHeaders(endpoint config.VLLMEndpoint) map[string]string {
	apiKey := strings.TrimSpace(endpoint.APIKey)
	if apiKey == "" {
		return nil
	}
	return map[string]string{"Authorization": "Bearer " + apiKey}
}
