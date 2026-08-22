package benchmark

import (
	"os"
	"path/filepath"
	"strings"
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

func TestLoadManifest_ShippedConfigResolvesCPUCI(t *testing.T) {
	manifest, err := LoadManifest(filepath.Join("..", "..", "config", "perf.yaml"))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	resolved, err := manifest.Resolve("cpu", "ci")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(resolved.Suites) != 6 {
		t.Fatalf("CPU CI suites = %d, want 6", len(resolved.Suites))
	}
	if len(resolved.Trends) != 4 {
		t.Fatalf("CPU CI trends = %d, want 4", len(resolved.Trends))
	}
	if resolved.Profile.Count != 3 || resolved.Profile.BenchTime != "500ms" {
		t.Fatalf("CPU CI profile = %+v", resolved.Profile)
	}
}

func TestManifestRejectsCapabilityMismatch(t *testing.T) {
	manifest := &Manifest{
		SchemaVersion: ManifestSchemaVersion,
		Environments: map[string]EnvironmentConfig{
			"cpu": {Kind: "cpu", Capabilities: []string{"host"}},
		},
		Profiles: map[string]ProfileConfig{
			"ci": {Suites: []string{"gpu-suite"}},
		},
		Suites: map[string]SuiteConfig{
			"gpu-suite": {
				Runner: "external", Module: ".", Command: []string{"run-gpu"},
				Environments: []string{"cpu"}, Requires: []string{"gpu"},
			},
		},
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("manifest structure should be valid: %v", err)
	}
	if _, err := manifest.Resolve("cpu", "ci"); err == nil {
		t.Fatal("Resolve should reject a missing GPU capability")
	}
}

func TestManifestRejectsNegativeProfileCount(t *testing.T) {
	manifest := &Manifest{
		SchemaVersion: ManifestSchemaVersion,
		Environments: map[string]EnvironmentConfig{
			"cpu": {Kind: "cpu", Capabilities: []string{"host"}},
		},
		Profiles: map[string]ProfileConfig{
			"ci": {Suites: []string{"core"}, Count: -1},
		},
		Suites: map[string]SuiteConfig{
			"core": {
				Runner: "go_benchmark", Module: ".", Packages: []string{"./..."},
				Benchmark: "BenchmarkCore", Environments: []string{"cpu"},
			},
		},
	}
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "non-negative") {
		t.Fatalf("negative count error = %v, want non-negative validation", err)
	}
}

func TestManifestRejectsInvalidTrendMetric(t *testing.T) {
	manifest := &Manifest{
		SchemaVersion: ManifestSchemaVersion,
		Environments: map[string]EnvironmentConfig{
			"cpu": {Kind: "cpu", Capabilities: []string{"host"}},
		},
		Profiles: map[string]ProfileConfig{"ci": {Suites: []string{"core"}}},
		Suites: map[string]SuiteConfig{
			"core": {
				Runner: "go_benchmark", Module: ".", Packages: []string{"./..."},
				Benchmark: "BenchmarkCore", Environments: []string{"cpu"},
			},
		},
		Trends: map[string]TrendConfig{
			"core": {
				Title: "Core", Suite: "core", Benchmark: "BenchmarkCore",
				XDimension: "items", Metric: "not-a-metric",
			},
		},
	}
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported metric") {
		t.Fatalf("invalid trend metric error = %v", err)
	}
}
