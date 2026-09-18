package config

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMemoryPersistenceBoundsAtConfigLoad(t *testing.T) {
	for _, source := range []ConfigSource{ConfigSourceFile, ConfigSourceKubernetes} {
		t.Run(string(source), func(t *testing.T) {
			testMemoryPersistenceBoundsAtConfigLoad(t, source)
		})
	}
}

func testMemoryPersistenceBoundsAtConfigLoad(t *testing.T, source ConfigSource) {
	t.Helper()
	for _, field := range []struct {
		name string
		max  int64
	}{
		{"timeout_seconds", MaxMemoryPersistenceDurationSeconds},
		{"concurrency", int64(MaxMemoryPersistenceConcurrency)},
		{"queue", int64(MaxMemoryPersistenceQueue)},
		{"shutdown_grace_seconds", MaxMemoryPersistenceDurationSeconds},
	} {
		for _, value := range []int64{-1, 0, 1, field.max, field.max + 1, math.MaxInt64} {
			for _, enabled := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%d/enabled=%t", field.name, value, enabled), func(t *testing.T) {
					payload := globalValidationDocument(t, source, fmt.Sprintf(`stores:
  memory:
    enabled: %t
    persistence:
      %s: %d
`, enabled, field.name, value))
					// Parsing is the startup/reload boundary: it must reject unsafe
					// values before a caller can construct the persistence runner,
					// including Kubernetes startup before any CRDs are available.
					path := filepath.Join(t.TempDir(), "config.yaml")
					if err := os.WriteFile(path, payload, 0o600); err != nil {
						t.Fatal(err)
					}
					for _, loader := range []struct {
						name string
						load func() (*RouterConfig, error)
					}{
						{"bytes", func() (*RouterConfig, error) { return ParseYAMLBytes(payload) }},
						{"file", func() (*RouterConfig, error) { return Parse(path) }},
					} {
						t.Run(loader.name, func(t *testing.T) {
							cfg, err := loader.load()
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
							if cfg.ConfigSource != source {
								t.Fatalf("config source = %q, want %q", cfg.ConfigSource, source)
							}
							if err := ValidateKubernetesConfigContracts(cfg); err != nil {
								t.Fatalf("valid persistence boundary rejected by Kubernetes validation: %v", err)
							}
						})
					}
				})
			}
		}
	}
}

func TestMemoryPersistenceBoundsAtKubernetesValidation(t *testing.T) {
	persistenceCases := []MemoryPersistenceConfig{
		{Concurrency: MaxMemoryPersistenceConcurrency + 1},
		{Queue: MaxMemoryPersistenceQueue + 1},
	}
	durationAboveMax := MaxMemoryPersistenceDurationSeconds + 1
	if durationAboveMax <= int64(^uint(0)>>1) {
		persistenceCases = append(persistenceCases,
			MemoryPersistenceConfig{TimeoutSeconds: int(durationAboveMax)},
			MemoryPersistenceConfig{ShutdownGraceSeconds: int(durationAboveMax)},
		)
	}
	for _, persistence := range persistenceCases {
		cfg := &RouterConfig{Memory: MemoryConfig{Persistence: persistence}}
		if err := ValidateKubernetesConfigContracts(cfg); err == nil || !strings.Contains(err.Error(), "global memory persistence") {
			t.Fatalf("unsafe persistence config must be rejected before runtime construction: %v", err)
		}
	}
}
