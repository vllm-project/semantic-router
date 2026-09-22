package benchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInventoryRejectsMissingAndUnregisteredMeasurements(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"benchmarks":["BenchmarkA","BenchmarkB"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	complete := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkA": {}, "BenchmarkB": {}}}
	missing := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkA": {}}}
	extra := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkA": {}, "BenchmarkB": {}, "BenchmarkUnknown": {}}}
	for _, tc := range []struct {
		name              string
		current, baseline *Baseline
		wantError         string
	}{
		{"complete", complete, complete, ""},
		{"current omission", missing, complete, "current missing"},
		{"both omit same case", missing, missing, "missing required"},
		{"missing baseline", complete, missing, "baseline missing"},
		{"unreviewed workload", extra, complete, "unregistered"},
		{"stale baseline", complete, extra, "unregistered"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateInventory(path, tc.current, tc.baseline)
			if tc.wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("got %v, want %q", err, tc.wantError)
			}
		})
	}
}

func TestInventoryRejectsDuplicateNames(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"benchmarks":["BenchmarkA","BenchmarkA"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	result := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkA": {}}}
	if err := ValidateInventory(path, result, result); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate inventory accepted: %v", err)
	}
}

func TestParserRejectsRepeatedResult(t *testing.T) {
	line := "BenchmarkA-8 1 10 ns/op 0 B/op 0 allocs/op\n"
	if _, err := ParseBenchOutput(strings.NewReader(line + line)); err == nil {
		t.Fatal("repeated benchmark silently replaced an earlier result")
	}
}

func TestBaselineDirectoryRejectsDuplicateResult(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"first.json", "second.json"} {
		writeBaselineFile(t, filepath.Join(dir, name), `{"benchmarks":{"BenchmarkA":{"ns_per_op":10}}}`)
	}
	if _, err := LoadBaselineDir(dir); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate suite baseline accepted: %v", err)
	}
}
