package benchmark

import (
	"fmt"
	"os"

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

// RegressionThreshold is the maximum tolerated ns/op regression, in percent.
type RegressionThreshold struct {
	MaxRegressionPercent float64 `yaml:"max_regression_percent"`
}

// BenchmarkRegressionThreshold maps a benchmark-name regexp to its maximum
// tolerated ns/op regression. Name is a human label for readability only.
type BenchmarkRegressionThreshold struct {
	Name                 string  `yaml:"name"`
	Pattern              string  `yaml:"pattern"`
	MaxRegressionPercent float64 `yaml:"max_regression_percent"`
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

	return &thresholds, nil
}
