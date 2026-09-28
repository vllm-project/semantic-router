package selection

import (
	"context"
	"sync"
	"testing"
)

func TestRegistry_ConcurrentReassignmentRaces(t *testing.T) {
	originalRegistry := GetGlobalRegistry()
	defer func() { SetGlobalRegistry(originalRegistry) }()

	const iterations = 2000
	selCtx := &SelectionContext{
		CandidateModels: createCandidateModels("m1"),
		QualityWeight:   0.5,
		CostWeight:      0.5,
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			registry := NewRegistry()
			registry.Register(MethodStatic, stubSelector{
				result: &SelectionResult{SelectedModel: "m1", Method: MethodStatic, Tier: TierSupported},
			})
			SetGlobalRegistry(registry)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			_ = GetSelector(MethodStatic)
			_, _ = Select(context.Background(), MethodStatic, selCtx)
		}
	}()
	wg.Wait()
}
