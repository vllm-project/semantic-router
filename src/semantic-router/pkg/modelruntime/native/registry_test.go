package native

import (
	"context"
	"reflect"
	"testing"
)

type mockAdapter struct {
	name Backend
}

func (m *mockAdapter) Name() Backend               { return m.name }
func (m *mockAdapter) Capabilities() CapabilitySet { return CapabilitySet{} }
func (m *mockAdapter) LoadModel(ctx context.Context, req LoadRequest) (ModelHandle, error) {
	return nil, nil
}
func (m *mockAdapter) UnloadModel(ctx context.Context, handle ModelHandle) error { return nil }
func (m *mockAdapter) Inference(ctx context.Context, handle ModelHandle, req InferenceRequest) (InferenceResponse, error) {
	return nil, nil
}
func (m *mockAdapter) Info() []ModelInfo {
	return []ModelInfo{
		{
			Backend:             m.name,
			Capabilities:        []Capability{CapabilityEmbedding},
			Family:              FamilyModernBERT,
			ModelName:           "mock-model",
			ModelPath:           "/path/to/mock",
			IsLoaded:            true,
			MaxSequenceLength:   1024,
			DefaultDimension:    768,
			ArtifactFormat:      ArtifactFullModel,
			Modality:            ModalityText,
			RequestedDimensions: 768,
			RuntimeDimensions:   768,
			RequestedLayers:     12,
			RuntimeLayers:       12,
			Provider:            "mock-provider",
			Device:              "mock-device",
			Version:             "1.0.0",
			FeatureFlags:        map[string]bool{"batching": true},
			UnsupportedReasons:  map[string]string{},
			RegistryMetadata:    map[string]string{"key": "value"},
		},
	}
}

func TestTaxonomyConstants(t *testing.T) {
	if BackendCandle != "candle" {
		t.Errorf("BackendCandle constant mismatch")
	}
	if CapabilityEmbedding != "embedding" {
		t.Errorf("CapabilityEmbedding constant mismatch")
	}
	if FamilyModernBERT != "ModernBERT" {
		t.Errorf("FamilyModernBERT constant mismatch")
	}
}

func TestRegistryThreadSafety(t *testing.T) {
	reg := NewRegistry()

	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(idx int) {
			_ = reg.Register(&mockAdapter{name: Backend("backend-" + string(rune(idx)))})
			reg.List()
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestRegistry_DuplicateRegistration(t *testing.T) {
	reg := NewRegistry()
	err1 := reg.Register(&mockAdapter{name: BackendCandle})
	if err1 != nil {
		t.Fatalf("expected nil, got %v", err1)
	}

	err2 := reg.Register(&mockAdapter{name: BackendCandle})
	if err2 == nil {
		t.Fatal("expected error on duplicate registration, got nil")
	}
}

func TestRegistry_NilEmptyRegistration(t *testing.T) {
	reg := NewRegistry()
	err1 := reg.Register(nil)
	if err1 == nil {
		t.Fatal("expected error on nil registration, got nil")
	}

	err2 := reg.Register(&mockAdapter{name: ""})
	if err2 == nil {
		t.Fatal("expected error on empty name registration, got nil")
	}
}

func TestRegistry_ListOrdering(t *testing.T) {
	reg := NewRegistry()
	_ = reg.Register(&mockAdapter{name: BackendRemote})
	_ = reg.Register(&mockAdapter{name: BackendONNX})
	_ = reg.Register(&mockAdapter{name: BackendCandle})
	_ = reg.Register(&mockAdapter{name: BackendOpenVINO})

	list := reg.List()
	if len(list) != 4 {
		t.Fatalf("expected 4 adapters, got %d", len(list))
	}

	expectedOrder := []Backend{BackendCandle, BackendONNX, BackendOpenVINO, BackendRemote}
	for i, expected := range expectedOrder {
		if list[i].Name() != expected {
			t.Errorf("expected %v at index %d, got %v", expected, i, list[i].Name())
		}
	}
}

func TestModelInfoDiscoveryShape(t *testing.T) {
	adapter := &mockAdapter{name: BackendCandle}
	info := adapter.Info()[0]

	if info.Backend != BackendCandle {
		t.Errorf("expected BackendCandle, got %v", info.Backend)
	}
	if !reflect.DeepEqual(info.Capabilities, []Capability{CapabilityEmbedding}) {
		t.Errorf("unexpected capabilities")
	}
	if info.ArtifactFormat != ArtifactFullModel {
		t.Errorf("unexpected artifact format")
	}
	if info.FeatureFlags["batching"] != true {
		t.Errorf("expected batching feature flag to be true")
	}
}
