package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// DefaultBackendMetricsPath is where engines expose Prometheus metrics unless
// a backend ref overrides it.
const DefaultBackendMetricsPath = "/metrics"

// CanonicalBackendMetrics overrides where a backend's engine exposes metrics.
// Port defaults to the backend endpoint's port and Path to /metrics.
type CanonicalBackendMetrics struct {
	Port int    `yaml:"port,omitempty"`
	Path string `yaml:"path,omitempty"`
}

// EndpointTelemetry is a runtime endpoint's telemetry source. Engine is empty
// when the endpoint reports none; MetricsPort zero means the endpoint's port.
type EndpointTelemetry struct {
	Engine      string `yaml:"engine,omitempty"`
	MetricsPort int    `yaml:"metrics_port,omitempty"`
	MetricsPath string `yaml:"metrics_path,omitempty"`
}

// Enabled reports whether the endpoint names an engine to collect from.
func (t EndpointTelemetry) Enabled() bool { return t.Engine != "" }

// BackendRefName is the backend_refs[].name this endpoint was materialized
// from: canonical import prefixes it with the model alias.
func (e VLLMEndpoint) BackendRefName() string {
	return strings.TrimPrefix(e.Name, e.Model+"_")
}

// TelemetryIdentity is the replica identity backend telemetry is keyed by.
func (e VLLMEndpoint) TelemetryIdentity() backendtelemetry.Identity {
	return backendtelemetry.Identity{
		Model:      e.Model,
		BackendRef: e.BackendRefName(),
		Engine:     backendtelemetry.Engine(e.Telemetry.Engine),
	}
}

// MetricsURL is the absolute URL of the endpoint's engine metrics.
func (e VLLMEndpoint) MetricsURL() string {
	scheme := strings.ToLower(strings.TrimSpace(e.Protocol))
	if scheme == "" {
		scheme = "http"
	}
	port := e.Telemetry.MetricsPort
	if port == 0 {
		port = e.Port
	}
	path := e.Telemetry.MetricsPath
	if path == "" {
		path = DefaultBackendMetricsPath
	}
	host := e.Address
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	return fmt.Sprintf("%s://%s:%d%s", scheme, host, port, path)
}

func validateCanonicalBackendTelemetry(modelName string, refs []CanonicalBackendRef) error {
	for index, ref := range refs {
		if err := validateBackendRefTelemetry(ref); err != nil {
			return fmt.Errorf("providers.models[%s].backend_refs[%d].%w", modelName, index, err)
		}
	}
	return nil
}

func validateBackendRefTelemetry(ref CanonicalBackendRef) error {
	if ref.Engine == "" {
		if ref.Metrics != nil {
			return fmt.Errorf("metrics requires engine")
		}
		return nil
	}
	if _, err := backendtelemetry.ParseEngine(ref.Engine); err != nil {
		return fmt.Errorf("engine: %w", err)
	}
	if ref.Metrics == nil {
		return nil
	}
	if ref.Metrics.Port < 0 || ref.Metrics.Port > 65535 {
		return fmt.Errorf("metrics.port %d must be within 1..65535", ref.Metrics.Port)
	}
	if path := ref.Metrics.Path; path != "" && (!strings.HasPrefix(path, "/") || strings.ContainsAny(path, "?# ")) {
		return fmt.Errorf("metrics.path %q must be an absolute path without query or fragment", path)
	}
	return nil
}

func instanceTelemetryFromBackendRef(ref CanonicalBackendRef) modelcatalog.InstanceTelemetry {
	telemetry := modelcatalog.InstanceTelemetry{Engine: ref.Engine}
	if ref.Metrics != nil {
		telemetry.MetricsPort = ref.Metrics.Port
		telemetry.MetricsPath = ref.Metrics.Path
	}
	return telemetry
}

func endpointTelemetryFromInstance(telemetry modelcatalog.InstanceTelemetry) EndpointTelemetry {
	return EndpointTelemetry{
		Engine: telemetry.Engine, MetricsPort: telemetry.MetricsPort, MetricsPath: telemetry.MetricsPath,
	}
}

func applyEndpointTelemetryToBackendRef(ref *CanonicalBackendRef, telemetry EndpointTelemetry) {
	ref.Engine = telemetry.Engine
	if telemetry.MetricsPort != 0 || telemetry.MetricsPath != "" {
		ref.Metrics = &CanonicalBackendMetrics{Port: telemetry.MetricsPort, Path: telemetry.MetricsPath}
	}
}
