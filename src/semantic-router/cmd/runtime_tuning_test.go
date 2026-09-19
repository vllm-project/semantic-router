package main

import (
	"os"
	"testing"
)

// tuningKeys are the native-math thread knobs applyBackendRuntimeTuningDefaults
// is responsible for defaulting.
var tuningKeys = []string{
	"OMP_NUM_THREADS",
	"MKL_NUM_THREADS",
	"OPENBLAS_NUM_THREADS",
	"RAYON_NUM_THREADS",
	"TOKENIZERS_PARALLELISM",
}

// clearTuningEnv unsets every tuning key for the test. t.Setenv is unusable
// here: the keys must be genuinely absent, not set to "".
func clearTuningEnv(t *testing.T) {
	t.Helper()
	for _, key := range append(tuningKeys, "EMBEDDING_BACKEND_OVERRIDE") {
		if previous, ok := os.LookupEnv(key); ok {
			t.Cleanup(func() { _ = os.Setenv(key, previous) })
		} else {
			t.Cleanup(func() { _ = os.Unsetenv(key) })
		}
		_ = os.Unsetenv(key)
	}
}

// The tuning used to apply only when EMBEDDING_BACKEND_OVERRIDE=candle, which
// left the common deployment oversubscribed.
func TestTuningDefaultsApplyWithoutBackendOverride(t *testing.T) {
	clearTuningEnv(t)

	applyBackendRuntimeTuningDefaults()

	if got := os.Getenv("RAYON_NUM_THREADS"); got != "1" {
		t.Fatalf("RAYON_NUM_THREADS = %q, want \"1\" with no backend override set", got)
	}
	for _, key := range tuningKeys {
		if os.Getenv(key) == "" {
			t.Errorf("%s was not defaulted", key)
		}
	}
}

// A single-request deployment may prefer intra-op parallelism, so an explicit
// value must survive.
func TestTuningDefaultsDoNotOverrideOperatorValues(t *testing.T) {
	clearTuningEnv(t)
	if err := os.Setenv("RAYON_NUM_THREADS", "8"); err != nil {
		t.Fatalf("seed RAYON_NUM_THREADS: %v", err)
	}

	applyBackendRuntimeTuningDefaults()

	if got := os.Getenv("RAYON_NUM_THREADS"); got != "8" {
		t.Fatalf("RAYON_NUM_THREADS = %q, want operator value \"8\" preserved", got)
	}
	// Keys the operator did not pin should still receive defaults.
	if got := os.Getenv("OMP_NUM_THREADS"); got != "1" {
		t.Fatalf("OMP_NUM_THREADS = %q, want \"1\"", got)
	}
}

func TestTuningDefaultsStillApplyForCandleBackend(t *testing.T) {
	clearTuningEnv(t)
	if err := os.Setenv("EMBEDDING_BACKEND_OVERRIDE", "candle"); err != nil {
		t.Fatalf("seed EMBEDDING_BACKEND_OVERRIDE: %v", err)
	}

	applyBackendRuntimeTuningDefaults()

	if got := os.Getenv("RAYON_NUM_THREADS"); got != "1" {
		t.Fatalf("RAYON_NUM_THREADS = %q, want \"1\" for the candle backend", got)
	}
}
