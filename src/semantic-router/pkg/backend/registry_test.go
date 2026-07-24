package backend

import (
	"context"
	"testing"
)

type fakeAdapter struct {
	kind Runtime
}

func (a fakeAdapter) Runtime() Runtime {
	return a.kind
}

func (a fakeAdapter) Collect(context.Context) ([]BackendTelemetry, error) {
	return []BackendTelemetry{{Identity: BackendIdentity{BackendRefName: "backend", ModelName: "model"}}}, nil
}

func TestRegistryCreatesRegisteredAdapter(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(RuntimeVLLM, func(AdapterConfig) (TelemetryAdapter, error) {
		return fakeAdapter{kind: RuntimeVLLM}, nil
	}); err != nil {
		t.Fatalf("Register returned error: %v", err)
	}
	if !registry.Has(RuntimeVLLM) {
		t.Fatal("expected registry to report vLLM adapter as registered")
	}

	adapter, err := registry.Create(RuntimeVLLM, AdapterConfig{})
	if err != nil {
		t.Fatalf("Create returned error: %v", err)
	}
	if adapter.Runtime() != RuntimeVLLM {
		t.Fatalf("Runtime = %q, want %q", adapter.Runtime(), RuntimeVLLM)
	}
}

func TestRegistryRejectsInvalidRegistrations(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register("", func(AdapterConfig) (TelemetryAdapter, error) {
		return fakeAdapter{}, nil
	}); err == nil {
		t.Fatal("expected empty runtime to be rejected")
	}
	if err := registry.Register(RuntimeATOM, nil); err == nil {
		t.Fatal("expected nil constructor to be rejected")
	}
	if _, err := registry.Create(RuntimeSGLang, AdapterConfig{}); err == nil {
		t.Fatal("expected unregistered adapter create to fail")
	}
}
