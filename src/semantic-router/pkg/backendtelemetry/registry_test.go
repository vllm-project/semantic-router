package backendtelemetry

import (
	"context"
	"errors"
	"testing"
)

type stubAdapter struct{ engine Engine }

func (a stubAdapter) Engine() Engine { return a.engine }

func (stubAdapter) Collect(context.Context) ([]Observation, error) { return nil, nil }

func constructorFor(engine Engine) Constructor {
	return func(AdapterConfig) (Adapter, error) { return stubAdapter{engine: engine}, nil }
}

func TestRegistryBuildsRegisteredAdapter(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(EngineVLLM, constructorFor(EngineVLLM)); err != nil {
		t.Fatal(err)
	}
	if !r.Has(EngineVLLM) || r.Has(EngineSGLang) {
		t.Fatal("Has does not reflect registrations")
	}
	adapter, err := r.New(EngineVLLM, AdapterConfig{})
	if err != nil || adapter.Engine() != EngineVLLM {
		t.Fatalf("New() = %v, %v", adapter, err)
	}
}

func TestRegistryRejectsBadRegistrations(t *testing.T) {
	r := NewRegistry()
	if err := r.Register("triton", constructorFor("triton")); err == nil {
		t.Fatal("registered an unknown engine")
	}
	if err := r.Register(EngineVLLM, nil); err == nil {
		t.Fatal("registered a nil constructor")
	}
	if err := r.Register(EngineVLLM, constructorFor(EngineVLLM)); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(EngineVLLM, constructorFor(EngineVLLM)); err == nil {
		t.Fatal("registered a second constructor for one engine")
	}
}

func TestRegistryNewFailures(t *testing.T) {
	r := NewRegistry()
	if _, err := r.New(EngineATOM, AdapterConfig{}); !errors.Is(err, ErrNoAdapter) {
		t.Fatalf("New(unregistered) = %v, want ErrNoAdapter", err)
	}
	if err := r.Register(EngineSGLang, constructorFor(EngineVLLM)); err != nil {
		t.Fatal(err)
	}
	if _, err := r.New(EngineSGLang, AdapterConfig{}); err == nil {
		t.Fatal("accepted an adapter built for another engine")
	}
	boom := errors.New("boom")
	if err := r.Register(EngineTensorRTLLM, func(AdapterConfig) (Adapter, error) { return nil, boom }); err != nil {
		t.Fatal(err)
	}
	if _, err := r.New(EngineTensorRTLLM, AdapterConfig{}); !errors.Is(err, boom) {
		t.Fatalf("New() = %v, want the constructor error", err)
	}
}
