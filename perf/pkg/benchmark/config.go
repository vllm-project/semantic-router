package benchmark

import (
	"fmt"
	"os"
	"regexp"

	"gopkg.in/yaml.v3"
)

// Config holds performance testing configuration
type Config struct {
	BenchmarkConfig BenchmarkConfigSection `yaml:"benchmark_config"`
	Profiling       ProfilingConfig        `yaml:"profiling"`
	Reporting       ReportingConfig        `yaml:"reporting"`
}

// BenchmarkConfigSection defines benchmark parameters
type BenchmarkConfigSection struct {
	Classification ClassificationConfig `yaml:"classification"`
	Cache          CacheConfig          `yaml:"cache"`
	E2E            E2EConfig            `yaml:"e2e"`
}

// ClassificationConfig defines classification benchmark parameters
type ClassificationConfig struct {
	BatchSizes       []int `yaml:"batch_sizes"`
	Iterations       int   `yaml:"iterations"`
	WarmupIterations int   `yaml:"warmup_iterations"`
}

// CacheConfig defines cache benchmark parameters
type CacheConfig struct {
	CacheSizes        []int   `yaml:"cache_sizes"`
	ConcurrencyLevels []int   `yaml:"concurrency_levels"`
	HitRatio          float64 `yaml:"hit_ratio"`
}

// E2EConfig defines E2E benchmark parameters
type E2EConfig struct {
	LoadPatterns []LoadPattern `yaml:"load_patterns"`
}

// LoadPattern defines a load testing pattern
type LoadPattern struct {
	Name     string `yaml:"name"`
	QPS      int    `yaml:"qps,omitempty"`
	StartQPS int    `yaml:"start_qps,omitempty"`
	EndQPS   int    `yaml:"end_qps,omitempty"`
	Duration string `yaml:"duration"`
}

// ProfilingConfig defines profiling settings
type ProfilingConfig struct {
	EnableCPU       bool   `yaml:"enable_cpu"`
	EnableMemory    bool   `yaml:"enable_memory"`
	EnableGoroutine bool   `yaml:"enable_goroutine"`
	OutputDir       string `yaml:"output_dir"`
}

// ReportingConfig defines reporting settings
type ReportingConfig struct {
	Formats     []string `yaml:"formats"`
	BaselineDir string   `yaml:"baseline_dir"`
}

// LoadConfig loads configuration from a YAML file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Set defaults
	if config.Profiling.OutputDir == "" {
		config.Profiling.OutputDir = "reports"
	}

	if config.Reporting.BaselineDir == "" {
		config.Reporting.BaselineDir = "testdata/baselines"
	}

	return &config, nil
}

// ThresholdsConfig holds performance threshold configuration
type ThresholdsConfig struct {
	ComponentBenchmarks ComponentBenchmarksThresholds `yaml:"component_benchmarks"`
	E2ETests            E2ETestsThresholds            `yaml:"e2e_tests"`
	ResourceLimits      ResourceLimitsThresholds      `yaml:"resource_limits"`
}

// ComponentBenchmarksThresholds defines regression thresholds for component
// benchmarks. Each entry matches a benchmark by name; Default applies to any
// benchmark no entry matches.
type ComponentBenchmarksThresholds struct {
	Default    RegressionThreshold            `yaml:"default"`
	Benchmarks []BenchmarkRegressionThreshold `yaml:"benchmarks"`
}

// RegressionThreshold holds the per-metric regression bounds for a benchmark.
//
// The gate blocks on allocs/op and B/op because they are hardware-independent
// (determined by the code path, not the CPU), so they compare cleanly across
// machines. ns/op is ADVISORY: MaxNsRegressionPercent is reported but never
// fails the gate, because absolute time depends on the runner's hardware.
type RegressionThreshold struct {
	MaxAllocsRegressionPercent float64 `yaml:"max_allocs_regression_percent"`
	MaxBytesRegressionPercent  float64 `yaml:"max_bytes_regression_percent"`
	MaxNsRegressionPercent     float64 `yaml:"max_ns_regression_percent,omitempty"`
}

// BenchmarkRegressionThreshold maps a benchmark-name regexp to its thresholds.
// Name is a human label for readability only.
type BenchmarkRegressionThreshold struct {
	Name                string `yaml:"name"`
	Pattern             string `yaml:"pattern"`
	RegressionThreshold `yaml:",inline"`
}

// E2ETestsThresholds defines thresholds for E2E tests
type E2ETestsThresholds struct {
	Throughput ThroughputThreshold `yaml:"throughput"`
	Latency    LatencyThreshold    `yaml:"latency"`
}

// ResourceLimitsThresholds defines resource limit thresholds
type ResourceLimitsThresholds struct {
	MaxMemoryMB   int     `yaml:"max_memory_mb"`
	MaxGoroutines int     `yaml:"max_goroutines"`
	MaxCPUPercent float64 `yaml:"max_cpu_percent"`
}

// ThroughputThreshold defines throughput thresholds
type ThroughputThreshold struct {
	MinSustainedQPS float64 `yaml:"min_sustained_qps"`
	MinSuccessRate  float64 `yaml:"min_success_rate"`
}

// LatencyThreshold defines latency thresholds
type LatencyThreshold struct {
	MaxP95Ms float64 `yaml:"max_p95_ms"`
	MaxP99Ms float64 `yaml:"max_p99_ms"`
}

// LoadThresholds loads threshold configuration from a YAML file
func LoadThresholds(path string) (*ThresholdsConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read thresholds file: %w", err)
	}

	var thresholds ThresholdsConfig
	if err := yaml.Unmarshal(data, &thresholds); err != nil {
		return nil, fmt.Errorf("failed to parse thresholds: %w", err)
	}

	// Reject invalid benchmark patterns up front. Otherwise a typo'd regexp is
	// skipped at match time and the benchmark silently falls back to the (looser)
	// default, quietly relaxing the gate with no signal.
	for _, b := range thresholds.ComponentBenchmarks.Benchmarks {
		if b.Pattern == "" {
			continue
		}
		if _, err := regexp.Compile(b.Pattern); err != nil {
			return nil, fmt.Errorf("invalid regexp for benchmark %q: %w", b.Name, err)
		}
	}

	return &thresholds, nil
}
