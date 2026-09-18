package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ValidateInventory requires both measurements to cover the selected inventory.
// An omitted result, an unreviewed new workload, or a stale baseline cannot pass.
func ValidateInventory(path string, current, baseline *Baseline) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read benchmark inventory: %w", err)
	}
	var inventory struct {
		Version    int      `json:"version"`
		Benchmarks []string `json:"benchmarks"`
	}
	if err := json.Unmarshal(data, &inventory); err != nil {
		return fmt.Errorf("decode benchmark inventory: %w", err)
	}
	if inventory.Version != 1 || len(inventory.Benchmarks) == 0 {
		return fmt.Errorf("benchmark inventory must be a nonempty version 1 inventory")
	}
	expected := &Baseline{Benchmarks: map[string]BenchmarkMetric{}}
	for _, name := range inventory.Benchmarks {
		if !strings.HasPrefix(name, "Benchmark") {
			return fmt.Errorf("invalid benchmark inventory entry %q", name)
		}
		if _, exists := expected.Benchmarks[name]; exists {
			return fmt.Errorf("duplicate benchmark inventory entry %s", name)
		}
		expected.Benchmarks[name] = BenchmarkMetric{}
	}
	for label, actual := range map[string]*Baseline{"current": current, "baseline": baseline} {
		if missing := MissingBenchmarks(actual, expected); len(missing) > 0 {
			return fmt.Errorf("%s missing required benchmarks: %s", label, strings.Join(missing, ", "))
		}
		if extra := UngatedBenchmarks(actual, expected); len(extra) > 0 {
			return fmt.Errorf("%s contains unregistered benchmarks: %s", label, strings.Join(extra, ", "))
		}
	}
	return nil
}
