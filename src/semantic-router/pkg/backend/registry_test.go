package backend

import (
	"context"
	"testing"
)

type fakeAdapter struct {
	kind EngineKind
}

func (a fakeAdapter) EngineKind() EngineKind {
	return a.kind
}

func (a fakeAdapter) Collect(context.Context) ([]BackendTelemetry, error) {
	return []BackendTelemetry{{Identity: BackendIdentity{BackendID: "backend", ModelName: "model"}}}, nil
}

func TestRegistryCreatesRegisteredAdapter(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(EngineKindVLLM, func(AdapterConfig) (TelemetryAdapter, error) {
		return fakeAdapter{kind: EngineKindVLLM}, nil
	}); err != nil {
		t.Fatalf("Register returned error: %v", err)
	}
	if !registry.Has(EngineKindVLLM) {
		t.Fatal("expected registry to report vLLM adapter as registered")
	}

	adapter, err := registry.Create(EngineKindVLLM, AdapterConfig{})
	if err != nil {
		t.Fatalf("Create returned error: %v", err)
	}
	if adapter.EngineKind() != EngineKindVLLM {
		t.Fatalf("EngineKind = %q, want %q", adapter.EngineKind(), EngineKindVLLM)
	}
}

func TestRegistryRejectsInvalidRegistrations(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register("", func(AdapterConfig) (TelemetryAdapter, error) {
		return fakeAdapter{}, nil
	}); err == nil {
		t.Fatal("expected empty engine kind to be rejected")
	}
	if err := registry.Register(EngineKindATOM, nil); err == nil {
		t.Fatal("expected nil constructor to be rejected")
	}
	if _, err := registry.Create(EngineKindSGLang, AdapterConfig{}); err == nil {
		t.Fatal("expected unregistered adapter create to fail")
	}
}
