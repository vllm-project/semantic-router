package config

import "testing"

const profilingConfigBaseYAML = `version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
`

func TestProfilingDefaultsWhenSectionOmitted(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(profilingConfigBaseYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned an error: %v", err)
	}

	profiling := cfg.Observability.Profiling
	if profiling.Enabled {
		t.Fatal("profiling must stay disabled unless the operator opts in")
	}
	if profiling.Port != DefaultProfilingPort {
		t.Fatalf("profiling port = %d, want %d", profiling.Port, DefaultProfilingPort)
	}
	if profiling.Bind != DefaultProfilingBind {
		t.Fatalf("profiling bind = %q, want %q", profiling.Bind, DefaultProfilingBind)
	}
}

func TestProfilingExplicitPortZeroSurvivesCanonicalDefaults(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(profilingConfigBaseYAML + `global:
  services:
    observability:
      profiling:
        enabled: true
        port: 0
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned an error: %v", err)
	}

	profiling := cfg.Observability.Profiling
	if !profiling.Enabled {
		t.Fatal("profiling enabled override was dropped")
	}
	if profiling.Port != 0 {
		t.Fatalf("profiling port = %d, want 0 so the listener takes an ephemeral port", profiling.Port)
	}
	if profiling.Bind != DefaultProfilingBind {
		t.Fatalf("profiling bind = %q, want the %q default to survive a port-only override", profiling.Bind, DefaultProfilingBind)
	}
}
