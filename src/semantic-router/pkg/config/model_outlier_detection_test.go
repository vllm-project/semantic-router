package config

import (
	"strings"
	"testing"
	"time"
)

func TestOutlierDetectionDefaults(t *testing.T) {
	for _, test := range []struct {
		name      string
		detection *DeploymentOutlierDetection
		want      ReplicaEjection
	}{
		{"unset", nil, ReplicaEjection{ConsecutiveTimeouts: 3, BaseEjection: 30 * time.Second, MaxEjection: 5 * time.Minute, MaxEjectionPercent: 50}},
		{"empty", &DeploymentOutlierDetection{}, ReplicaEjection{ConsecutiveTimeouts: 3, BaseEjection: 30 * time.Second, MaxEjection: 5 * time.Minute, MaxEjectionPercent: 50}},
		{"off", &DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(0)}, ReplicaEjection{ConsecutiveTimeouts: 0, BaseEjection: 30 * time.Second, MaxEjection: 5 * time.Minute, MaxEjectionPercent: 50}},
		{"the base sets the default maximum", &DeploymentOutlierDetection{BaseEjectionMs: 5000}, ReplicaEjection{ConsecutiveTimeouts: 3, BaseEjection: 5 * time.Second, MaxEjection: 50 * time.Second, MaxEjectionPercent: 50}},
		{"the default maximum is at most an hour", &DeploymentOutlierDetection{BaseEjectionMs: 600_000}, ReplicaEjection{ConsecutiveTimeouts: 3, BaseEjection: 10 * time.Minute, MaxEjection: time.Hour, MaxEjectionPercent: 50}},
		{"every field", &DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(5), BaseEjectionMs: 1000, MaxEjectionMs: 4000, MaxEjectionPercent: 25}, ReplicaEjection{ConsecutiveTimeouts: 5, BaseEjection: time.Second, MaxEjection: 4 * time.Second, MaxEjectionPercent: 25}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := (ModelDeployment{OutlierDetection: test.detection}).Ejection(); got != test.want {
				t.Fatalf("got %+v, want %+v", got, test.want)
			}
		})
	}
}

func TestOutlierDetectionIsValidated(t *testing.T) {
	replicas := []ModelReplica{{Device: "cpu"}, {Device: "cpu"}}
	managed := func(detection DeploymentOutlierDetection) ModelDeployment {
		return ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "/models/decision", Replicas: replicas, OutlierDetection: &detection}
	}
	for _, test := range []struct {
		name       string
		deployment ModelDeployment
		want       string
	}{
		{"every field in range", managed(DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(0), BaseEjectionMs: 1000, MaxEjectionMs: 3_600_000, MaxEjectionPercent: 100}), ""},
		{"attached replicas", ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Replicas: []ModelReplica{{Endpoint: "http://worker-1:8100"}, {Endpoint: "http://worker-2:8100"}}, OutlierDetection: &DeploymentOutlierDetection{}}, ""},
		{"a negative threshold", managed(DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(-1)}), "outlier_detection.consecutive_timeouts must be between 0 and 100"},
		{"a threshold over 100", managed(DeploymentOutlierDetection{ConsecutiveTimeouts: intPtr(101)}), "outlier_detection.consecutive_timeouts must be between 0 and 100"},
		{"a negative base", managed(DeploymentOutlierDetection{BaseEjectionMs: -1}), "outlier_detection.base_ejection_ms must be between 0 and 3600000"},
		{"a maximum over an hour", managed(DeploymentOutlierDetection{MaxEjectionMs: 3_600_001}), "outlier_detection.max_ejection_ms must be between 0 and 3600000"},
		{"a maximum below the base", managed(DeploymentOutlierDetection{BaseEjectionMs: 10_000, MaxEjectionMs: 5000}), "outlier_detection.max_ejection_ms must not be below base_ejection_ms (10000)"},
		{"a maximum below the default base", managed(DeploymentOutlierDetection{MaxEjectionMs: 1000}), "must not be below base_ejection_ms (30000)"},
		{"a share over 100", managed(DeploymentOutlierDetection{MaxEjectionPercent: 101}), "outlier_detection.max_ejection_percent must be between 0 and 100"},
		{"a remote model", ModelDeployment{Provider: OpenAIChatProvider, Endpoint: "http://chat:8000/v1", Model: "chat", OutlierDetection: &DeploymentOutlierDetection{}}, "outlier_detection and public_name belong to model_runtime"},
		{"a systemone model", ModelDeployment{Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "d3", OutlierDetection: &DeploymentOutlierDetection{}}, "outlier_detection and public_name belong to model_runtime"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := test.deployment.WithDefaults().validate()
			if test.want == "" && err != nil || test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("got %v, want %q", err, test.want)
			}
		})
	}
}

func TestOutlierDetectionLoadsFromTheModelCatalog(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
global:
  model_catalog:
    deployments:
      pool:
        provider: model_runtime
        artifact: /models/decision
        replicas: [{device: rocm:0}, {device: rocm:1}]
        outlier_detection:
          consecutive_timeouts: 5
          base_ejection_ms: 10000
          max_ejection_ms: 60000
          max_ejection_percent: 25
`))
	if err != nil {
		t.Fatal(err)
	}
	want := ReplicaEjection{ConsecutiveTimeouts: 5, BaseEjection: 10 * time.Second, MaxEjection: time.Minute, MaxEjectionPercent: 25}
	if got := cfg.ModelDeployments["pool"].Ejection(); got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}
