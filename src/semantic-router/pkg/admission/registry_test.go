package admission

import (
	"context"
	"testing"
)

func TestRegistryFor(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowShed)
	registry := NewRegistry(map[string]Admissioner{"prompt_guard": gate})

	if registry.For("prompt_guard") != Admissioner(gate) {
		t.Fatal("configured deployment must return its gate")
	}
	if _, ok := registry.For("domain_classifier").(Noop); !ok {
		t.Fatal("unconfigured deployment must return Noop")
	}
	var nilRegistry *Registry
	if _, ok := nilRegistry.For("prompt_guard").(Noop); !ok {
		t.Fatal("nil registry must return Noop")
	}
}

func TestRegistryNoopAdmitsEverything(t *testing.T) {
	registry := NewRegistry(nil)
	ticket, err := registry.For("feedback_detector").Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ticket()
}
