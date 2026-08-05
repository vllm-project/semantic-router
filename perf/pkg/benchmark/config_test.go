package benchmark

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadThresholds_InvalidPatternFails guards against a typo'd benchmark
// pattern silently loosening the gate: an uncompilable regexp would be skipped
// at match time, dropping the benchmark to the (looser) default with no signal.
// LoadThresholds must reject it up front instead.
func TestLoadThresholds_InvalidPatternFails(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "thresholds.yaml")
	content := `component_benchmarks:
  default:
    max_allocs_regression_percent: 10
  benchmarks:
    - name: broken
      pattern: "([unclosed"
      max_allocs_regression_percent: 5
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := LoadThresholds(path); err == nil {
		t.Fatal("expected LoadThresholds to reject an invalid benchmark pattern")
	}
}

// TestLoadThresholds_ShippedConfigLoads ensures the validation does not reject
// the real shipped config.
func TestLoadThresholds_ShippedConfigLoads(t *testing.T) {
	if _, err := LoadThresholds(filepath.Join("..", "..", "config", "thresholds.yaml")); err != nil {
		t.Fatalf("shipped thresholds.yaml should load: %v", err)
	}
}
