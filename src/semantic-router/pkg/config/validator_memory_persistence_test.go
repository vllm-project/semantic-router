package config

import (
	"fmt"
	"strings"
	"testing"
)

func TestMemoryPersistenceBoundsAtConfigLoad(t *testing.T) {
	for _, field := range []struct {
		name string
		max  int
	}{
		{"concurrency", MaxMemoryPersistenceConcurrency},
		{"queue", MaxMemoryPersistenceQueue},
	} {
		for _, value := range []int{-1, 0, 1, field.max, field.max + 1, int(^uint(0) >> 1)} {
			for _, enabled := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%d/enabled=%t", field.name, value, enabled), func(t *testing.T) {
					payload := fmt.Sprintf(`version: v0.3
providers:
  defaults:
    model: test-model
routing:
  modelCards:
    - name: test-model
global:
  stores:
    memory:
      enabled: %t
      persistence:
        %s: %d
`, enabled, field.name, value)
					// Parsing is the startup/reload boundary: it must reject unsafe
					// values before a caller can construct the persistence runner.
					cfg, err := ParseYAMLBytes([]byte(payload))
					if value < 0 || value > field.max {
						want := fmt.Sprintf("persistence %s must be between 0 and %d", field.name, field.max)
						if err == nil || !strings.Contains(err.Error(), want) || cfg != nil {
							t.Fatalf("expected config rejection containing %q, got config=%v, err=%v", want, cfg, err)
						}
						return
					}
					if err != nil {
						t.Fatalf("valid persistence boundary rejected: %v", err)
					}
					if err := ValidateKubernetesConfigContracts(cfg); err != nil {
						t.Fatalf("valid persistence boundary rejected by Kubernetes validation: %v", err)
					}
				})
			}
		}
	}
}

func TestMemoryPersistenceBoundsAtKubernetesValidation(t *testing.T) {
	for _, persistence := range []MemoryPersistenceConfig{
		{Concurrency: MaxMemoryPersistenceConcurrency + 1},
		{Queue: MaxMemoryPersistenceQueue + 1},
	} {
		cfg := &RouterConfig{Memory: MemoryConfig{Persistence: persistence}}
		if err := ValidateKubernetesConfigContracts(cfg); err == nil || !strings.Contains(err.Error(), "global memory persistence") {
			t.Fatalf("unsafe persistence config must be rejected before runtime construction: %v", err)
		}
	}
}
